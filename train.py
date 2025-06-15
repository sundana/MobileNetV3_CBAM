"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import data_setup
import engine
from models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from models.proposed_model import (
    MobileNetV3_Large_CBAM_16,
    MobileNetV3_Large_CBAM_32,
    MobileNetV3_Small_CBAM_16,
    MobileNetV3_Small_CBAM_32,
)
import utils
from torchvision import transforms


def start_training(
    model_name: str,
    num_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
):
    # Setup directories
    data_path = "data/"

    # Setup target device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create transforms
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Create DataLoaders with help from data_setup.py
    train_dataloader, val_dataloader, test_dataloader, class_names = (
        data_setup.create_dataloader(
            data_path=data_path, transform=data_transform, batch_size=batch_size
        )
    )

    # Model selection using dictionary mapping
    model_map = {
        "mobilenetv3_small": MobileNetV3_Small,
        "proposed_large_16": MobileNetV3_Large_CBAM_16,
        "proposed_large_32": MobileNetV3_Large_CBAM_32,
        "proposed_small_16": MobileNetV3_Small_CBAM_16,
        "proposed_small_32": MobileNetV3_Small_CBAM_32,
    }

    # Get model class from map or default to MobileNetV3_Large
    model_class = model_map.get(model_name, MobileNetV3_Large)
    model = model_class(num_classes=len(class_names)).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Show training config
    print(f"Model: {model.__class__.__name__}")
    print(f"Epoch: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Optimizer: {optimizer}")
    print(f"Device: {device}")

    # Start training with help from engine.py
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=num_epochs,
        device=device,
    )

    # Save the model with help from utils.py
    utils.save_model(
        model=model,
        target_dir="checkpoints",
        model_name="05_going_modular_script_mode_tinyvgg_model.pth",
    )

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for train model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="mobilenetv3_large",
        choices=[
            "mobilenetv3_small",
            "mobilenetv3_largeproposed_large_16",
            "proposed_large_32",
            "proposed_small_16",
            "proposed_small_32",
        ],
        help="name of the model",
    )
    args = parser.parse_args()
    model_name = args.model
    start_training(model_name)
