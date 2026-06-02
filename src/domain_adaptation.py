"""
Domain Adaptation methods for bridging PlantVillage -> PlantDoc domain shift.

Implements:
1. Gradient Reversal Layer (Ganin & Lempitsky, 2015) for adversarial DA
2. Domain classifier head attached to the feature extractor
3. CutMix and MixUp transforms for strong augmentation
4. Training loops for DANN and augmented training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import time


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation.
    
    During forward pass: identity function.
    During backward pass: multiplies gradients by -lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    """MLP domain classifier attached to feature space."""
    def __init__(self, in_features: int, hidden_dim: int = 256, num_domains: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x):
        return self.net(x)


class MobileNetV3FeatureExtractor(nn.Module):
    """Wraps a MobileNetV3 model, exposing features before the classifier.

    The feature vector is the output of the GlobalAveragePooling layer
    (after the final conv2 + bn2 + hs2 + avg_pool, before linear3).

    For the custom MobileNetV3 in this project, the penultimate features
    are the 1280-dim vector after linear3+bn3+hs3 (or 960/576-dim after avg_pool
    depending on which layer we tap).
    """
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x):
        # Run the backbone but stop before the final classifier (linear4)
        # Custom MobileNetV3 forward path:
        # conv1 > bn1 > hs1 > bneck > conv2 > bn2 > hs2 > avg_pool > flatten > linear3 > bn3 > hs3 > linear4
        out = self.backbone.hs1(self.backbone.bn1(self.backbone.conv1(x)))
        out = self.backbone.bneck(out)
        out = self.backbone.hs2(self.backbone.bn2(self.backbone.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        features = self.backbone.hs3(self.backbone.bn3(self.backbone.linear3(out)))
        # features shape: (batch, 1280)
        return features


class DANNWrapper(nn.Module):
    """Wraps a MobileNetV3 model for Domain-Adversarial training.

    Composed of:
    - feature_extractor: backbone up to the penultimate layer
    - label_classifier: the final linear layer (linear4)
    - domain_classifier: adversarial head predicting source/target domain
    - GRL: gradient reversal between features and domain classifier
    """
    def __init__(self, backbone: nn.Module, num_classes: int, feature_dim: int = 1280):
        super().__init__()
        self.feature_extractor = MobileNetV3FeatureExtractor(backbone, feature_dim)
        self.label_classifier = backbone.linear4  # reuse final classifier
        self.domain_classifier = DomainClassifier(feature_dim, num_domains=2)

    def forward(self, x, lambda_: float = 1.0):
        features = self.feature_extractor(x)
        class_logits = self.label_classifier(features)
        # Apply GRL for domain classification
        reversed_features = GradientReversalLayer.apply(features, lambda_)
        domain_logits = self.domain_classifier(reversed_features)
        return class_logits, domain_logits


# ---------------------------------------------------------------------------
# CutMix and MixUp transforms (operate on batches of images + labels)
# ---------------------------------------------------------------------------

def cutmix_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation to a batch.
    
    Returns:
        mixed_images, labels_a, labels_b, lam
        where labels_a and labels_b are the original labels for the two images,
        and lam is the mixing proportion.
    """
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    rand_index = torch.randperm(batch_size, device=images.device)

    # Generate bounding box
    _, _, H, W = images.shape
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cut_w = int(W * np.sqrt(1. - lam))
    cut_h = int(H * np.sqrt(1. - lam))
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    # Adjust lambda based on actual cut area
    lam = 1. - ((x2 - x1) * (y2 - y1) / (W * H))

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    labels_a = labels
    labels_b = labels[rand_index]

    return mixed_images, labels_a, labels_b, lam


def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """Apply MixUp augmentation to a batch."""
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    rand_index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    labels_a = labels
    labels_b = labels[rand_index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp/CutMix loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# DANN Training Loop
# ---------------------------------------------------------------------------

def train_epoch_dann(
    dann_model: DANNWrapper,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    class_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    lambda_schedule: bool = True,
) -> Tuple[float, float, float, float]:
    """Train one epoch of DANN.

    The domain adaptation lambda increases gradually:
        lambda_p = 2 / (1 + exp(-10 * p)) - 1
    where p = epoch / total_epochs.

    Returns:
        (class_loss, domain_loss, source_acc, domain_acc)
    """
    dann_model.train()

    p = epoch / total_epochs
    lambda_ = (2. / (1. + np.exp(-10. * p)) - 1.) if lambda_schedule else 1.0

    total_class_loss = 0.0
    total_domain_loss = 0.0
    total_source_correct = 0
    total_samples = 0
    total_domain_correct = 0
    total_domain_samples = 0

    # Zip source and target loaders; cycle target if shorter
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    n_batches = max(len(source_loader), len(target_loader))

    pbar = tqdm(range(n_batches), desc=f"DANN Epoch {epoch + 1}", leave=False, ncols=100)

    for _ in pbar:
        try:
            src_images, src_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            src_images, src_labels = next(source_iter)

        try:
            tgt_images, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_images, _ = next(target_iter)

        src_images = src_images.to(device)
        src_labels = src_labels.to(device)
        tgt_images = tgt_images.to(device)

        batch_size_src = src_images.size(0)
        batch_size_tgt = tgt_images.size(0)

        # Create domain labels: 0 = source, 1 = target
        domain_src = torch.zeros(batch_size_src, dtype=torch.long, device=device)
        domain_tgt = torch.ones(batch_size_tgt, dtype=torch.long, device=device)

        # Forward pass on source
        src_class_logits, src_domain_logits = dann_model(src_images, lambda_)
        class_loss = class_criterion(src_class_logits, src_labels)

        # Forward pass on target (no class labels needed)
        tgt_domain_logits_from_model = dann_model.feature_extractor(tgt_images)
        reversed_tgt = GradientReversalLayer.apply(tgt_domain_logits_from_model, lambda_)
        tgt_domain_logits = dann_model.domain_classifier(reversed_tgt)

        # Domain classification loss
        domain_loss_src = F.cross_entropy(src_domain_logits, domain_src)
        domain_loss_tgt = F.cross_entropy(tgt_domain_logits, domain_tgt)
        domain_loss = domain_loss_src + domain_loss_tgt

        # Total loss
        loss = class_loss + domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_class_loss += class_loss.item()
        total_domain_loss += domain_loss.item()

        src_correct = (src_class_logits.argmax(dim=1) == src_labels).sum().item()
        total_source_correct += src_correct
        total_samples += batch_size_src

        all_domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
        all_domain_labels = torch.cat([domain_src, domain_tgt], dim=0)
        domain_correct = (all_domain_logits.argmax(dim=1) == all_domain_labels).sum().item()
        total_domain_correct += domain_correct
        total_domain_samples += batch_size_src + batch_size_tgt

        pbar.set_postfix({
            "Cls Loss": f"{class_loss.item():.4f}",
            "Dom Loss": f"{domain_loss.item():.4f}",
            "Src Acc": f"{src_correct / batch_size_src:.3f}",
            "Lambda": f"{lambda_:.3f}",
        })

    avg_class_loss = total_class_loss / n_batches
    avg_domain_loss = total_domain_loss / n_batches
    source_acc = total_source_correct / total_samples
    domain_acc = total_domain_correct / total_domain_samples

    return avg_class_loss, avg_domain_loss, source_acc, domain_acc


def validate_dann(
    dann_model: DANNWrapper,
    val_loader: torch.utils.data.DataLoader,
    class_criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate DANN model on target domain (classification only)."""
    dann_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            class_logits, _ = dann_model(images, lambda_=0.0)
            loss = class_criterion(class_logits, labels)

            total_loss += loss.item()
            total_correct += (class_logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(val_loader), total_correct / total_samples
