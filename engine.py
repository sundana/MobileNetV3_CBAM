"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from typing import Tuple, Dict, List
from tqdm import tqdm
import time
from utils import EarlyStopping, TrainingLogger


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Create progress bar for training batches
    batch_pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Training",
        leave=False,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    # Loop through data loader data batches
    for batch, (X, y) in batch_pbar:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        batch_acc = (y_pred_class == y).sum().item() / len(y_pred)
        train_acc += batch_acc

        # Update progress bar with current metrics
        current_avg_loss = train_loss / (batch + 1)
        current_avg_acc = train_acc / (batch + 1)
        batch_pbar.set_postfix(
            {
                "Loss": f"{current_avg_loss:.4f}",
                "Acc": f"{current_avg_acc:.4f}",
                "Batch Loss": f"{loss.item():.4f}",
                "Batch Acc": f"{batch_acc:.4f}",
            }
        )

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def valid_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validation dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the validation data.
    device: A target device to compute on (e.g. "cuda", "mps" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0

    # Create progress bar for validation batches
    batch_pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Validation",
        leave=False,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in batch_pbar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            batch_acc = (val_pred_labels == y).sum().item() / len(val_pred_labels)
            val_acc += batch_acc

            # Update progress bar with current metrics
            current_avg_loss = val_loss / (batch + 1)
            current_avg_acc = val_acc / (batch + 1)
            batch_pbar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Acc": f"{current_avg_acc:.4f}",
                    "Batch Loss": f"{loss.item():.4f}",
                    "Batch Acc": f"{batch_acc:.4f}",
                }
            )

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = val_loss / len(dataloader)
    test_acc = val_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    patience: int = 5,
    early_stopping: bool = False,
    min_delta: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    enable_live_plot: bool = True,
    scheduler=None,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch model through train_step() and valid_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.
    Supports early stopping functionality to prevent overfitting.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        valid_dataloader: A DataLoader instance for the model to be validated on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        patience: Number of epochs with no improvement after which training will be stopped.
        early_stopping: Boolean indicating whether to use early stopping.
        min_delta: Minimum change in monitored value to qualify as improvement.
        checkpoint_dir: Directory to save model checkpoints for early stopping.
        enable_live_plot: Whether to enable real-time plotting during training.
        scheduler: Optional learning rate scheduler.

    Returns:
        A dictionary of training and validation metrics for each epoch:
        {train_loss: [...], train_acc: [...], valid_loss: [...], valid_acc: [...]}
    """
    # Setup tracking
    start_time = time.time()
    results = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    model.to(device)

    # Initialize early stopping if enabled
    early_stopper = None
    if early_stopping:
        early_stopper = EarlyStopping(
            patience=patience,
            delta=min_delta,
            save_best_model=True,
            checkpoint_dir=checkpoint_dir,
        )

    # Initialize enhanced training logger
    logger = TrainingLogger(
        model_name=model.__class__.__name__,
        log_dir="results/training",
        enable_live_plot=enable_live_plot,
        plot_every=1,  # Update plots every epoch
    )

    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="Overall Progress", ncols=120, position=0)

    # Training loop
    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None

        # Train and validate for current epoch
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        valid_loss, valid_acc = valid_step(
            model=model, dataloader=valid_dataloader, loss_fn=loss_fn, device=device
        )

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Update learning rate scheduler if provided
        if scheduler:
            # Handle different scheduler types
            if hasattr(scheduler, "step"):
                if "ReduceLROnPlateau" in scheduler.__class__.__name__:
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        # Log results with enhanced information
        logger.log_epoch(
            epoch + 1,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            epoch_time=epoch_time,
            learning_rate=current_lr,
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)

        # Update main progress bar with comprehensive metrics
        epoch_pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Train Acc": f"{train_acc:.4f}",
                "Val Loss": f"{valid_loss:.4f}",
                "Val Acc": f"{valid_acc:.4f}",
                "Best Val Acc": f"{logger.best_val_acc:.4f}",
                "LR": f"{current_lr:.2e}" if current_lr else "N/A",
                "Time": f"{epoch_time:.1f}s",
            }
        )

        # Print detailed progress (optional, can be disabled for cleaner output)
        print(f"\nEpoch: {epoch + 1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Valid - Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}")
        print(
            f"  Time: {epoch_time:.2f}s, LR: {current_lr:.2e}"
            if current_lr
            else f"  Time: {epoch_time:.2f}s"
        )
        if logger.best_val_acc == valid_acc:
            print(f"  🎉 New best validation accuracy!")

        # Check early stopping
        if early_stopper:
            early_stopper(model=model, val_loss=valid_loss, epoch=epoch + 1)
            if early_stopper.early_stop:
                print(f"\n🔴 Early stopping triggered at epoch {epoch + 1}")
                break

    # Finalize training
    training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"📊 Final Results:")
    print(f"   • Total time: {int(training_time // 60)}m {int(training_time % 60)}s")
    print(
        f"   • Best validation accuracy: {logger.best_val_acc:.4f} (Epoch {logger.best_epoch})"
    )
    print(f"   • Best validation loss: {logger.best_val_loss:.4f}")
    print(f"   • Total epochs: {epoch + 1}")

    # Save results
    logger.save_to_csv()
    logger.save_final_plots()

    return results
