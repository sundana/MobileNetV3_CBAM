import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import evaluate_model
from torchvision import transforms
from data_setup import create_dataloader
import os
from dotenv import load_dotenv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument("-w", "--weight", help="Model weight")
parser.add_argument("-d", "--device", help="Device")

args = parser.parse_args()

model = args.model
weight = args.weight
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

match model:
    case "proposed_model_large":
        from my_models.mobilenetv3 import MobileNetV3_Large
        model = MobileNetV3_Large(num_classes=num_classes)
        model.load_state_dict(torch.load(f=f'/Users/firmansyahsundana/Documents/Study/computer science/tesis/code/checkpoints/{weight}.pth', map_location=device))
    case "proposed_model_small":
        from my_models.mobilenetv3 import MobileNetV3_Small
        model = MobileNetV3_Small(num_classes=num_classes)
        model.load_state_dict(torch.load(f=f'/Users/firmansyahsundana/Documents/Study/computer science/tesis/code/checkpoints/{weight}.pth', map_location=device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

cm, performance_table, final_loss = evaluate_model(model, criterion, test_loader, class_names, device=device)
from evaluations import measure_throughput_latency

avg_latency, throughput = measure_throughput_latency(model, test_loader, device=device)

print(f"Average Latency: {avg_latency:.4f} seconds per batch")
print(f"Throughput: {throughput:.2f} samples per second")
