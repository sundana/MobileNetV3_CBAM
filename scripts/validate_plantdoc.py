import sys
import os
import torch

# Fix Unicode emoji printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
from functools import partial
import numpy as np
from tqdm import tqdm

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.models.baselines import get_mobilenet_v2, get_shufflenet_v2
from src.config import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLANTDOC_DIR
from src.utils import evaluate_model

# Mapping from PlantDoc directory names to PlantVillage class names
PLANTDOC_PV_MAPPING = {
    'Apple leaf': 'Apple___healthy',
    'Apple rust leaf': 'Apple___Cedar_apple_rust',
    'Apple Scab Leaf': 'Apple___Apple_scab',
    'Blueberry leaf': 'Blueberry___healthy',
    'Cherry leaf': 'Cherry___healthy',
    'grape leaf': 'Grape___healthy',
    'grape leaf black rot': 'Grape___Black_rot',
    'Peach leaf': 'Peach___healthy',
    'Raspberry leaf': 'Raspberry___healthy',
    'Squash Powdery mildew leaf': 'Squash___Powdery_mildew',
    'Strawberry leaf': 'Strawberry___healthy',
    'Tomato Early blight leaf': 'Tomato___Early_blight',
    'Tomato leaf': 'Tomato___healthy',
    'Tomato leaf bacterial spot': 'Tomato___Bacterial_spot',
    'Tomato leaf late blight': 'Tomato___Late_blight',
    'Tomato leaf mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf': 'Tomato___Leaf_Mold',
    'Tomato Septoria leaf spot': 'Tomato___Septoria_leaf_spot',
    'Tomato two spotted spider mites leaf': 'Tomato___Spider_mites Two-spotted_spider_mite'
}

PV_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Raspberry___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

class PlantDocMappedDataset(Dataset):
    def __init__(self, root_dir, mapping, pv_classes, transform=None):
        self.root_dir = root_dir
        self.mapping = mapping
        self.pv_classes = pv_classes
        self.transform = transform
        
        self.samples = []
        print(f"📦 Filtering PlantDoc dataset for mapped classes...")
        for pd_class, pv_class in mapping.items():
            pd_path = os.path.join(root_dir, pd_class)
            if not os.path.exists(pd_path):
                print(f"⚠️  Warning: PD class path not found: {pd_path}")
                continue
            
            pv_idx = pv_classes.index(pv_class)
            class_samples = 0
            for img_name in os.listdir(pd_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(pd_path, img_name), pv_idx))
                    class_samples += 1
            # print(f"   • {pd_class} -> {pv_class}: {class_samples} images")

        print(f"✅ Total mapped samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a dummy image or handle appropriately
            return torch.zeros((3, 224, 224)), label

def main():
    parser = argparse.ArgumentParser(description="Validate model on PlantDoc dataset")
    parser.add_argument("-m", "--model", help="Model name", required=True)
    parser.add_argument("-w", "--weight", help="Model weight file name", required=True)
    parser.add_argument("-d", "--device", help="Device (cuda/cpu)", default="auto")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🔍 Starting PlantDoc validation on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PlantDocMappedDataset(
        root_dir=PLANTDOC_DIR,
        mapping=PLANTDOC_PV_MAPPING,
        pv_classes=PV_CLASSES,
        transform=transform
    )

    if len(dataset) == 0:
        print("❌ No samples found in PlantDoc dataset.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model_map = {
        "mobilenetv3_small": partial(MobileNetV3_Small, attention_type='se'),
        "mobilenetv3_large": partial(MobileNetV3_Large, attention_type='se'),
        "proposed_large_16": partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=16),
        "proposed_large_32": partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=32),
        "proposed_small_16": partial(MobileNetV3_Small, attention_type='cbam', reduction_ratio=16),
        "proposed_small_32": partial(MobileNetV3_Small, attention_type='cbam', reduction_ratio=32),
        "mobilenetv2": get_mobilenet_v2,
        "shufflenetv2": get_shufflenet_v2,
    }

    model_factory = model_map.get(args.model)
    if not model_factory:
        print(f"❌ Unknown model: {args.model}")
        return

    model = model_factory(num_classes=len(PV_CLASSES)).to(device)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.weight}.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # Handle key mismatch if necessary (e.g., 'module' vs 'attention_module')
    new_state_dict = {}
    for k, v in state_dict.items():
        if ".se.se." in k:
            k = k.replace(".se.se.", ".attention_module.se.")
        if ".module.channel_attention.fc1." in k:
            k = k.replace(".module.channel_attention.fc1.", ".attention_module.channel_attention.se.0.")
        if ".module.channel_attention.fc2." in k:
            k = k.replace(".module.channel_attention.fc2.", ".attention_module.channel_attention.se.2.")
        if ".module.spatial_attention.conv." in k:
            k = k.replace(".module.spatial_attention.conv.", ".attention_module.spatial_attention.conv.")
        if "bneck" in k and ".module." in k:
            k = k.replace(".module.", ".attention_module.")
        new_state_dict[k] = v
    
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"⚠️  First load attempt failed: {e}")
        print("🔄 Attempting to load without key mapping...")
        model.load_state_dict(state_dict)
    
    print(f"✅ Loaded weights from {checkpoint_path}")

    # Run evaluation
    results_dir = os.path.join(RESULTS_DIR, "plantdoc_validation", args.model)
    os.makedirs(results_dir, exist_ok=True)

    cm, metrics, final_loss = evaluate_model(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        test_loader=dataloader,
        class_names=PV_CLASSES,
        device=device,
        results_dir=results_dir
    )

    print(f"\n✨ Validation on PlantDoc finished!")
    print(f"📊 Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
