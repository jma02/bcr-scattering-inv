# coding=utf-8
# vim: sw=4 et tw=100
"""
PyTorch training callback for MNN

Converted from Keras callback to PyTorch training utilities.
"""

import time
import os
import numpy as np
import torch
from typing import Optional, Callable, List, Tuple, Union, Dict, Any


class SaveBestModel:
    """Save the best model during training.
    
    This class provides functionality similar to Keras callbacks but for PyTorch.
    It monitors training and validation metrics and saves the best model.
    
    Args:
        filename: Path to save the model file
        check_result: Function to evaluate model performance
        verbose: Verbosity level (0 or 1)
        period: Interval (number of epochs) between checks
        output: Output function (default: print)
        patience: Number of epochs to wait before early stopping
        model_for_save: Optional separate model to save
        test_weight: Weight for combining train/test errors (0-1)
        reduce_lr: Whether to reduce learning rate on plateau
        min_lr: Minimum learning rate
        patience_lr: Patience for learning rate reduction
        factor: Factor by which to reduce learning rate
    """
    
    def __init__(self, 
                 filename: str,
                 check_result: Optional[Callable] = None,
                 verbose: int = 1,
                 period: int = 1,
                 output: Callable = print,
                 patience: int = 10000,
                 model_for_save: Optional[torch.nn.Module] = None,
                 test_weight: float = 0.5,
                 reduce_lr: bool = False,
                 min_lr: Optional[float] = None,
                 patience_lr: Optional[int] = None,
                 factor: float = 0.5):
        
        self.filename = filename
        self.check_result = check_result
        self.verbose = verbose
        self.period = period
        self.output = output
        self.patience = patience
        self.model_for_save = model_for_save
        self.test_weight = max(min(test_weight, 1), 0)
        self.reduce_lr = reduce_lr
        self.min_lr = min_lr
        self.patience_lr = patience_lr or patience // 2
        self.factor = factor
        
        # State variables
        self.best_epoch = 0
        self.best_epoch_update = 0
        self.epochs_since_last_save = 0
        self.best_err_train = float('inf')
        self.best_err_test = float('inf')
        self.best_err_train_max = float('inf')
        self.best_err_test_max = float('inf')
        self.best_err_var_train = float('inf')
        self.best_err_var_test = float('inf')
        self.start_time = time.time()
        self.stop_epoch = 0
        self.err_history: List[Tuple[int, float, float]] = []
        self.should_stop = False
        
        # Current optimizer reference (set during training)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    
    def on_train_begin(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Called at the beginning of training."""
        self.optimizer = optimizer
        self.compare_with_best_model(model, -1)
        
        if self.reduce_lr:
            self.patience_lr = min(self.patience_lr, self.patience)
            if self.min_lr is None:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.min_lr = current_lr / 10
                self.min_lr = max(self.min_lr, 1e-5)
            
            self.output(f"Initial LR = {self.optimizer.param_groups[0]['lr']:.6f}")
            self.output(f"Min LR = {self.min_lr:.6f}")
            self.output(f"Patience = {self.patience}")
            self.output(f"Patience LR = {self.patience_lr}")
    
    def on_epoch_end(self, model: torch.nn.Module, epoch: int) -> bool:
        """Called at the end of each epoch.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.compare_with_best_model(model, epoch)
        
        # Handle learning rate reduction
        if self.reduce_lr and (epoch + 1 - self.best_epoch_update) >= self.patience_lr:
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr > self.min_lr + 1e-7:
                new_lr = current_lr * self.factor
                new_lr = max(new_lr, self.min_lr)
                
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                self.output(f'\nEpoch {epoch + 1:05d}: Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}.')
                
                # Reload best model with proper device handling
                try:
                    device = next(model.parameters()).device
                    model.load_state_dict(torch.load(self.filename, map_location=device))
                    self.best_epoch = epoch + 1
                    self.best_epoch_update = epoch + 1
                except Exception as e:
                    self.output(f'Warning: Could not load best model: {e}')
        
        # Check for early stopping
        if (epoch + 1 - self.best_epoch_update) >= self.patience:
            self.should_stop = True
            self.stop_epoch = epoch
            self.output(f'Early stopped at epoch {self.stop_epoch} with patience {self.patience}')
            return True
        
        return False
    
    def compare_with_best_model(self, model: torch.nn.Module, epoch: int):
        """Compare current model with best model and save if better."""
        if self.check_result is None:
            return
        
        t1 = time.time()
        
        # Evaluate model
        with torch.no_grad():
            model.eval()
            err_train, err_test = self.check_result(model)
            model.train()
        
        # Record error history
        mean_train_err = np.mean(err_train)
        mean_test_err = np.mean(err_test)
        self.err_history.append([epoch + 1, mean_train_err, mean_test_err])
        
        # Compute combined error
        tw = self.test_weight
        err_old = self.best_err_train * (1 - tw) + self.best_err_test * tw
        err_new = mean_train_err * (1 - tw) + mean_test_err * tw
        
        # Check if this is the best model OR if no model exists yet
        model_exists = os.path.exists(self.filename)
        is_better = err_old > err_new and abs(err_new / err_old - 1) > 1e-3
        
        if not model_exists or is_better:
            self.best_epoch = epoch + 1
            self.best_epoch_update = epoch + 1
            self.best_err_train = mean_train_err
            self.best_err_test = mean_test_err
            self.best_err_train_max = np.max(err_train)
            self.best_err_test_max = np.max(err_test)
            self.best_err_var_train = np.var(err_train)
            self.best_err_var_test = np.var(err_test)
            
            # Save model
            model_to_save = self.model_for_save if self.model_for_save is not None else model
            torch.save(model_to_save.state_dict(), self.filename)
            
            if self.verbose >= 1:
                elapsed = time.time() - t1
                self.output(f'Epoch {epoch + 1:05d}: Saved model with '
                           f'train_err={mean_train_err:.6f}, '
                           f'test_err={mean_test_err:.6f} '
                           f'(took {elapsed:.2f}s)')
        
        elif self.verbose >= 2:
            elapsed = time.time() - t1
            self.output(f'Epoch {epoch + 1:05d}: '
                       f'train_err={mean_train_err:.6f}, '
                       f'test_err={mean_test_err:.6f} '
                       f'(took {elapsed:.2f}s)')


def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                num_epochs: int,
                device: torch.device,
                callback: Optional[SaveBestModel] = None,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                grad_clip: Optional[float] = None,
                weight_decay: float = 0.0) -> Dict[str, Any]:
    """Train a PyTorch model with optional callbacks.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Number of epochs to train
        device: Device to run training on
        callback: Optional callback for model saving/early stopping
        scheduler: Optional learning rate scheduler
        grad_clip: Optional gradient clipping threshold (default: None)
        weight_decay: Manual L2 regularization weight (default: 0.0)
    
    Returns:
        Dictionary with training history
    """
    model.to(device)
    model.train()
    
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    
    # Initialize callback if provided
    if callback:
        callback.on_train_begin(model, optimizer)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add L2 regularization if weight_decay is specified
            if weight_decay > 0:
                l2_reg = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += weight_decay * l2_reg
            
            loss.backward()
            
            # Apply gradient clipping if specified
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epochs'].append(epoch + 1)
        
        # Update scheduler
        if scheduler:
            if hasattr(scheduler, 'step') and 'metrics' in scheduler.step.__code__.co_varnames:
                # ReduceLROnPlateau needs a metric
                scheduler.step(avg_val_loss)
            else:
                # StepLR and others don't need a metric
                scheduler.step()
        
        # Handle callback
        should_stop = False
        if callback:
            should_stop = callback.on_epoch_end(model, epoch)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}')
        
        if should_stop:
            break
    
    return history
