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


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test data")
    parser.add_argument("-m", "--model", help="Model name", required=True)
    parser.add_argument("-w", "--weight", help="Model weight file name", required=True)
    parser.add_argument("-d", "--device", help="Device (cuda/cpu)", default="auto")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--results_dir", default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🔍 Starting evaluation on device: {device}")

    load_dotenv()
    data_path = os.environ.get("DATA_PATH")

    if not data_path:
        print("❌ DATA_PATH not found in environment variables")
        return

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("📁 Loading datasets...")
    train_loader, val_loader, test_loader, class_names = create_dataloader(
        data_path=data_path,
        transform=transform,
        batch_size=args.batch_size,
    )

    num_classes = len(class_names)
    print(f"📊 Found {num_classes} classes: {class_names}")

    # Load model based on model type
    print(f"🤖 Loading model: {args.model}")

    model = None
    checkpoint_path = f"checkpoints/{args.weight}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    try:
        if args.model == "proposed_model_large":
            from models.mobilenetv3 import MobileNetV3_Large

            model = MobileNetV3_Large(num_classes=num_classes)
        elif args.model == "proposed_model_small":
            from models.mobilenetv3 import MobileNetV3_Small

            model = MobileNetV3_Small(num_classes=num_classes)
        elif args.model == "mobilenetv3_large":
            from models.mobilenetv3 import MobileNetV3_Large

            model = MobileNetV3_Large(num_classes=num_classes)
        elif args.model == "mobilenetv3_small":
            from models.mobilenetv3 import MobileNetV3_Small

            model = MobileNetV3_Small(num_classes=num_classes)
        else:
            print(f"❌ Unknown model type: {args.model}")
            return

        # Load model weights
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
            print(
                f"✅ Loaded model checkpoint from epoch {state_dict.get('epoch', 'unknown')}"
            )
        else:
            model.load_state_dict(state_dict)
            print("✅ Loaded model weights")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    print("\n🔬 Starting comprehensive evaluation...")

    # Enhanced evaluation with better visualization
    cm, performance_metrics, final_loss = evaluate_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        results_dir=args.results_dir,
    )

    # Performance measurement
    try:
        from evaluations import measure_throughput_latency

        print("\n⚡ Measuring performance...")
        avg_latency, throughput = measure_throughput_latency(
            model, test_loader, device=device
        )
        print(f"⏱️  Average Latency: {avg_latency:.4f} seconds per batch")
        print(f"🚀 Throughput: {throughput:.2f} samples per second")
    except ImportError:
        print("⚠️  Performance measurement module not available")
    except Exception as e:
        print(f"⚠️  Error measuring performance: {e}")

    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
