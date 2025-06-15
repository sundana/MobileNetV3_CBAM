"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import data_setup
import engine
from models.mobilenetv3 import MobileNetV3_Small
import utils
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001

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
data_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
    data_path=data_path,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = MobileNetV3_Small(num_classes=len(class_names)).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             valid_dataloader=val_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(
    model=model,
    target_dir="checkpoints",
    model_name="05_going_modular_script_mode_tinyvgg_model.pth"
)
