import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os

# Define EarlyStopping class with checkpoint saving
class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_best_model=True, checkpoint_dir='checkpoints'):
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
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
                self.save_checkpoint(model, epoch)  # Save the model checkpoint
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
                self.save_checkpoint(model, epoch)  # Save the model checkpoint
            self.counter = 0

    def save_checkpoint(self, model, epoch):
        """Save the model checkpoint with the current epoch."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_best_model(self, model):
        """Load the best model's weights."""
        if self.save_best_model:
            model.load_state_dict(self.best_model_wts)



def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, patience=5, checkpoint_dir='checkpoints'):
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
    early_stopping = EarlyStopping(patience=patience, delta=0, save_best_model=True, checkpoint_dir=checkpoint_dir)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
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

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

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



def evaluate_model(model, criterion, test_loader, class_names, device="mpu"):
  model = model.to(device)
  model.eval()
  all_preds = []
  all_labels = []

  total_loss = 0.0
  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      _, preds = torch.max(outputs, 1)
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
      total_loss += loss.item() * inputs.size(0)

  # Calculate avg loss
  final_loss = total_loss / len(test_loader)
  print(f"Final Loss: {final_loss:.4f}")

  cm = confusion_matrix(all_labels, all_preds)
  accuracy = accuracy_score(all_labels, all_preds)
  precision = precision_score(all_labels, all_preds, average='weighted')
  recall = recall_score(all_labels, all_preds, average='weighted')
  f1 = f1_score(all_labels, all_preds, average='weighted')

  # Plot confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()

  # Display performance matrix
  print(f"Accuracy: {accuracy:.2f}")
  print(f"Precision: {precision:.2f}")
  print(f"Recall: {recall:.2f}")
  print(f"F1 Score: {f1:.4f}")

  # Create a table of performance metrics
  performance_table = {
      'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
      'Value': [f"{accuracy:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.4f}"]
  }

  return cm, performance_table, final_loss



def save_model(model, optimizer, num_epochs, final_loss):
   # Save a full checkpoint with additional details
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # Optional if you need it
        'epoch': num_epochs,
        'loss': final_loss,  # Replace final_loss with your final loss value
    }
    torch.save(checkpoint, f'J:\\tesis\\checkpoints\\{dt_string.replace("/", "-").replace(":", "-")}.pth')



def calculate_inference_time(model, test_loader, device="cpu"):
    """
    Calculate the inference time of a PyTorch model.

    Parameters:
    - model: torch.nn.Module, the PyTorch model to evaluate
    - input_tensor: torch.Tensor, a sample input tensor matching the model's input shape
    - device: str, device to run the inference ('cuda' or 'cpu')

    Returns:
    - float, average inference time in milliseconds
    """
   
    # if torch.cuda.is_available():
    #     device = device
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    # else:
    #     device = "cpu"

    model = model.to(device)

    model.eval()
    # Warm-up runs to stabilize CUDA performance
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break  # Run warm-up on just one batch

     # Measure inference time across the dataset
    times = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Return the average inference time per batch
    return sum(times) / len(times)
