import time
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os 
import argparse
from utils import plot_training_history, train_model
from dotenv import load_dotenv
from data_setup import create_dataloader
from torchvision import transforms
from evaluations import measure_throughput_latency

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_best_model=True, checkpoint_dir='checkpoints'):
        """
        Args:
        patience (int): How many epochs to wait after the last improvement before stopping.
        delta (float): Minimum change to qualify as an improvement.
        save_best_model (bool): Whether to save the best model based on validation loss.
        checkpoint_dir (str): Directory where to save the best model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.save_best_model = save_best_model
        self.checkpoint_dir = checkpoint_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        """
        Call the early stopping function to decide whether to stop training.
        
        Args:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): The model to save the best weights for.
        epoch (int): Current epoch number to include in the checkpoint file name.
        """
        score = -val_loss  
        if self.best_score is None:
            self.best_score = score
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  
                self.save_checkpoint(model, epoch)  
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  
                self.save_checkpoint(model, epoch)  
            self.counter = 0

    def save_checkpoint(self, model, epoch):
        """Save the model checkpoint with the current epoch."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_best_model(self, model):
        """Load the best model's weights."""
        if self.save_best_model:
            model.load_state_dict(self.best_model_wts)


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, patience=5, checkpoint_dir='checkpoints'):
    start_time = time.time()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    model.to(device)
    
    early_stopping = EarlyStopping(patience=patience, delta=0, save_best_model=True, checkpoint_dir=checkpoint_dir)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        early_stopping(val_loss, model, epoch)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    early_stopping.load_best_model(model)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in: {total_time // 60:.0f}m {total_time % 60:.0f}s")

    return history



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("-e", "--epoch", help="Num. of epoch")
    parser.add_argument("-d", "--device", help="Device")

    args = parser.parse_args()

    model = args.model
    num_epochs = int(args.epoch)
    device = args.device

    load_dotenv()
    data_path = os.environ.get('DATA_PATH')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (or the input size for your model)
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    train_loader, val_loader, test_loader, class_names = create_dataloader(
        data_path = data_path,
        transform = transform,
        batch_size = 64,
    )

    num_classes = len(class_names)

    if model == "proposed_model_large":
        from my_models.mobilenetv3 import MobileNetV3_Large
        model = MobileNetV3_Large(num_classes=num_classes)
    elif model == "proposed_model_small":
        from my_models.mobilenetv3 import MobileNetV3_Small
        model = MobileNetV3_Small(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device=device)
    plot_training_history(history, save_dir='plots')

    