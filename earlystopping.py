import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_best_model=True):
        """
        Args:
        patience (int): How many epochs to wait after the last improvement before stopping.
        delta (float): Minimum change to qualify as an improvement.
        save_best_model (bool): Whether to save the best model based on validation loss.
        """
        self.patience = patience
        self.delta = delta
        self.save_best_model = save_best_model
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        """
        Call the early stopping function to decide whether to stop training.
        
        Args:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): The model to save the best weights for.
        """
        score = -val_loss  # We minimize loss, so we negate it to maximize the score
        if self.best_score is None:
            self.best_score = score
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
            self.counter = 0

    def load_best_model(self, model):
        """Load the best model's weights."""
        if self.save_best_model:
            model.load_state_dict(self.best_model_wts)
