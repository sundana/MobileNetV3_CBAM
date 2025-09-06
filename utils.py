import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from datetime import datetime
import os
import pandas as pd
import csv


# Define EarlyStopping class with checkpoint saving
class EarlyStopping:
    def __init__(
        self, patience=5, delta=0, save_best_model=True, checkpoint_dir="checkpoints"
    ):
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
        self.best_checkpoint_path = None
        self.best_epoch = 0

        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        """
        Call the early stopping function to decide whether to stop training.

        Args:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): The model to save the best weights for.
        epoch (int): Current epoch number to include in the checkpoint file name.
        """
        score = -val_loss  # We minimize loss, so we negate it to maximize the score
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
                self.save_checkpoint(model, epoch)  # Save the model checkpoint
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
                self.save_checkpoint(model, epoch)  # Save the model checkpoint
            self.counter = 0

    def save_checkpoint(self, model, epoch):
        """Save the best model checkpoint, overwriting previous best model."""
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            os.remove(self.best_checkpoint_path)  # Remove previous best checkpoint

        model_name = model.__class__.__name__
        self.best_checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{model_name}_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), self.best_checkpoint_path)
        print(f"Best checkpoint saved to {self.best_checkpoint_path}")

    def load_best_model(self, model):
        """Load the best model's weights."""
        if self.save_best_model and self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
            print(f"Loaded best model weights from epoch {self.best_epoch}")


class TrainingLogger:
    def __init__(self, model_name=str, log_dir="results/training"):
        """
        Initialize the training logger.

        Args:
            model_name (str): The name of the model being trained
            log_dir (str): Directory to save log files
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.model_name = model_name if model_name else "model"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, f"{self.model_name}_training_log_{timestamp}.csv"
        )
        self.results = []

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log results for one epoch."""
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        self.results.append(epoch_data)

    def save_to_csv(self):
        """Save all results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.log_file, index=False)
        print(f"Training log saved to {self.log_file}")


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    patience=5,
    checkpoint_dir="checkpoints",
):
    start_time = time.time()

    # Lists to store loss and accuracy for each epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    model.to(device)

    # Initialize EarlyStopping object
    early_stopping = EarlyStopping(
        patience=patience, delta=0, save_best_model=True, checkpoint_dir=checkpoint_dir
    )

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy and loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # Calculate average loss and accuracy for training
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Validation phase
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

        # Calculate average loss and accuracy for validation
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Call early stopping
        early_stopping(val_loss, model, epoch)

        # If early stopping triggered, break the loop
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Load the best model after training
    early_stopping.load_best_model(model)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in: {total_time // 60:.0f}m {total_time % 60:.0f}s")

    return history


def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.ylim(0, 100)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_model(
    model, criterion, test_loader, class_names, device, results_dir="./results"
):
    """
    Evaluate the model on the test dataset and calculate performance metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): Loss function.
        test_loader (DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names for the confusion matrix.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
        results_dir (str): Directory to save the confusion matrix plot.

    Returns:
        tuple: Confusion matrix, performance metrics table, and final loss.
    """
    start_time = time.time()
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0

    # Evaluate the model
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item() * inputs.size(0)

    # Calculate average loss
    final_loss = total_loss / len(test_loader.dataset)
    print(f"Final Loss: {final_loss:.4f}")

    # Calculate performance metrics
    cm = confusion_matrix(all_labels, all_preds)
    metrics = calculate_metrics(all_labels, all_preds)

    # Plot and save confusion matrix
    save_confusion_matrix(cm, class_names, model.__class__.__name__, results_dir)

    # Display performance metrics
    print_metrics(metrics)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Inference completed in: {total_time // 60:.0f}m {total_time % 60:.0f}s")

    return cm, metrics, final_loss


def calculate_metrics(labels, preds):
    """
    Calculate evaluation metrics.

    Args:
        labels (list): True labels.
        preds (list): Predicted labels.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


def save_confusion_matrix(cm, class_names, model_name, results_dir):
    """
    Save the confusion matrix as a heatmap, both overall and per-class.

    Args:
        cm (ndarray): Confusion matrix.
        class_names (list): List of class names.
        model_name (str): Name of the model.
        results_dir (str): Directory to save the plot.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save the overall confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plot_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Overall confusion matrix saved to {plot_path}")
    plt.close()

    # Create and save per-class confusion matrices
    for i, class_name in enumerate(class_names):
        # Create binary confusion matrix for this class
        # True positives, false negatives
        # False positives, true negatives
        binary_cm = np.zeros((2, 2), dtype=int)

        # True positives (predicted as this class and is this class)
        binary_cm[0, 0] = cm[i, i]

        # False negatives (predicted as another class but is this class)
        binary_cm[0, 1] = np.sum(cm[i, :]) - cm[i, i]

        # False positives (predicted as this class but is another class)
        binary_cm[1, 0] = np.sum(cm[:, i]) - cm[i, i]

        # True negatives (predicted as another class and is another class)
        binary_cm[1, 1] = (
            np.sum(cm) - binary_cm[0, 0] - binary_cm[0, 1] - binary_cm[1, 0]
        )

        # Create the plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            binary_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted as " + class_name, "Predicted as others"],
            yticklabels=["Actually " + class_name, "Actually others"],
        )
        plt.title(f"Confusion Matrix for {class_name}")

        # Save the plot
        class_plot_path = os.path.join(
            results_dir,
            f"{model_name}_confusion_matrix_{class_name.replace(' ', '_')}.png",
        )
        plt.savefig(class_plot_path, bbox_inches="tight")
        print(f"Class confusion matrix saved to {class_plot_path}")
        plt.close()

    # Display the overall confusion matrix when calling the function
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def print_metrics(metrics):
    """
    Print evaluation metrics.

    Args:
        metrics (dict): Dictionary of evaluation metrics.
    """
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")


def save_model(model, optimizer, num_epochs, final_loss):
    # Save a full checkpoint with additional details
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # Optional if you need it
        "epoch": num_epochs,
        "loss": final_loss,  # Replace final_loss with your final loss value
    }
    torch.save(
        checkpoint,
        f"checkpoints/{model.__class__.__name__}-{dt_string.replace('/', '-').replace(':', '-')}.pth",
    )


def calculate_inference_time(model, test_loader, device="cpu"):
    """
    Calculate the inference time of a PyTorch model.

    Parameters:
    - model: torch.nn.Module, the PyTorch model to evaluate
    - test_loader: DataLoader, the test dataset loader
    - device: str, device to run the inference ('cuda' or 'cpu')

    Returns:
    - float, average inference time in milliseconds
    """
    model = model.to(device)
    model.eval()

    # Warm-up runs to stabilize CUDA performance
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break  # Run warm-up on just one batch

    # Measure inference time across the dataset with a progress bar
    times = []
    with torch.no_grad():
        for inputs, _ in tqdm(
            test_loader, desc="Calculating Inference Time", unit="batch"
        ):
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Return the average inference time per batch
    return sum(times) / len(times)


def export_confusion_matrix_to_csv(
    cm, class_names, save_path="./csv", filename="confusion_matrix.csv"
):
    """
    Export a confusion matrix to a CSV file with proper row and column labels.

    Parameters:
    -----------
    cm : numpy.ndarray
        The confusion matrix to export
    class_names : list
        List of class names for labeling rows and columns
    save_path : str, optional
        Directory path where the CSV file will be saved
    filename : str, optional
        Name of the CSV file

    Returns:
    --------
    str
        Path to the saved CSV file
    """
    import os
    import numpy as np
    import pandas as pd

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create a DataFrame with proper labeling
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Add row labels for clarity (actual classes)
    cm_df.index.name = "Actual"

    # Add a custom header for the first column (predicted classes)
    full_path = os.path.join(save_path, filename)

    # Save to CSV
    cm_df.to_csv(full_path)
    print(f"Confusion matrix saved to {full_path}")

    return full_path
