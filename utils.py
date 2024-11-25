import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    start_time = time.time()

    # Lists to store loss and accuracy for each epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

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
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()



def evaluate_model(model, criterion, test_loader, class_names, device="cuda"):
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
