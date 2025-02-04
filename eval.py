import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import evaluate_model
from torchvision import transforms
from data_setup import create_dataloader
import os
from dotenv import load_dotenv


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

loaded_mobilenetv3_cbam_large = MobileNetV3WithCBAM(mode='large', num_classes=num_classes)
loaded_mobilenetv3_cbam_large.load_state_dict(torch.load(f='/Users/firmansyahsundana/Documents/Study/computer science/tesis/code/checkpoints/proposed_model_large.pth', map_location="mps"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenetv3_cbam_large.parameters(), lr=0.001)

cm, performance_table, final_loss = evaluate_model(loaded_mobilenetv3_cbam_large, criterion, test_loader, class_names, device=device)
from evaluations import measure_throughput_latency

avg_latency, throughput = measure_throughput_latency(loaded_mobilenetv3_cbam_large, test_loader, device='cpu')

print(f"Average Latency: {avg_latency:.4f} seconds per batch")
print(f"Throughput: {throughput:.2f} samples per second")