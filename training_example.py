"""
Example script demonstrating the enhanced training visualization features.
This script shows how to use the improved training engine with real-time plotting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Import our enhanced training functions
from engine import train
from utils import training_dashboard_summary, quick_plot_metrics, compare_training_runs


def create_dummy_model(num_classes=10):
    """Create a simple CNN model for demonstration."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )


def main():
    """
    Demonstration of enhanced training visualization.

    This function shows how to:
    1. Use the enhanced training function with real-time plotting
    2. Generate training summaries
    3. Compare multiple training runs
    """

    print("🚀 Enhanced Training Visualization Demo")
    print("=" * 50)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    # Create a simple dataset (CIFAR-10-like dummy data)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # For demo purposes, we'll use a small synthetic dataset
    # In practice, you would load your actual dataset here
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 3, 32, 32),  # 1000 samples, 3 channels, 32x32
        torch.randint(0, 10, (1000,)),  # 10 classes
    )

    # Split dataset
    train_size = int(0.8 * len(dummy_dataset))
    val_size = len(dummy_dataset) - train_size
    train_dataset, val_dataset = random_split(dummy_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = create_dummy_model(num_classes=10)

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("\n🎯 Starting training with enhanced visualization...")
    print("Features enabled:")
    print("  ✅ Real-time plotting")
    print("  ✅ Progress bars with metrics")
    print("  ✅ Learning rate tracking")
    print("  ✅ Training time monitoring")
    print("  ✅ Best model highlighting")
    print("  ✅ Automatic plot saving")

    # Train the model with enhanced visualization
    results = train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        epochs=10,  # Small number for demo
        device=device,
        early_stopping=True,
        patience=3,
        enable_live_plot=True,  # Enable real-time plotting
        scheduler=scheduler,
    )

    print("\n📊 Training completed! Check the results folder for:")
    print("  • Real-time training plots")
    print("  • Training log CSV files")
    print("  • Training summary statistics")

    # Demonstrate utility functions
    print("\n🔧 Utility Functions Demo:")

    # Show training dashboard summary
    try:
        print("\n1. Training Dashboard Summary:")
        training_dashboard_summary("results/training")
    except Exception as e:
        print(f"Dashboard summary error: {e}")

    # Example of how to load and re-plot from CSV
    print("\n2. To re-plot training history from saved CSV:")
    print("   quick_plot_metrics('path/to/training_log.csv', 'results/plots')")

    print("\n3. To compare multiple training runs:")
    print("   compare_training_runs(['log1.csv', 'log2.csv'], ['Model1', 'Model2'])")

    print("\n🎉 Demo completed! Your training visualization is now enhanced!")


if __name__ == "__main__":
    main()
