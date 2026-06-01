import os
from typing import Optional

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

# For Windows compatibility, default NUM_WORKERS to 0
NUM_WORKERS = 0

from src.config import DATA_DIR

class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def create_dataloader(
    data_path: str = DATA_DIR,
    train_transform: transforms.Compose = None,
    test_transform: transforms.Compose = None,
    batch_size: int = 64,
    num_workers: Optional[int] = NUM_WORKERS
):
    # Load base dataset without transform to get raw PIL images
    base_dataset = datasets.ImageFolder(root=data_path, transform=None)

    train_size = int(0.7 * len(base_dataset))  # 70% for training
    val_size = int(0.15 * len(base_dataset))   # 15% for validation
    test_size = len(base_dataset) - train_size - val_size  # Remaining 15% for testing

    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size]
    )

    # Wrap subsets with their respective transforms
    train_dataset = SubsetDataset(train_subset, train_transform)
    val_dataset = SubsetDataset(val_subset, test_transform)
    test_dataset = SubsetDataset(test_subset, test_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers if num_workers else 0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers if num_workers else 0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers if num_workers else 0
    )

    class_names = base_dataset.classes
    
    return train_dataloader, val_dataloader, test_dataloader, class_names



if __name__ == "__main__":
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    create_dataloader(
        data_path='data', 
        batch_size=64, 
        num_workers=2, 
        train_transform=train_transform,
        test_transform=test_transform
    )
    