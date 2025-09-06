import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

NUM_WORKERS = os.cpu_count()

def create_dataloader(
    data_path: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.15 * len(dataset))   # 15% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = test_dataloader.dataset.dataset.classes
    
    return train_dataloader, val_dataloader, test_dataloader, class_names
