import time
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os 

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




if __name__ == "__main__":
    pass

