# train_updated.py
# Based on train.py (Combined v1.0 - Merged, Corrected, and Reconstructed)
# --- Modifications ---
# 1. embedding_dim set to 32
# 2. hidden_dim set to 64
# 3. epochs set to 20
# 4. early stopping patience set to 3
# --------------------

import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset # Added TensorDataset
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau # For LR scheduling
import time # For timing outputs
import copy # For saving best model state

# --- Device Detection (from v1.0 snippet) ---
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() # Optional: For Apple Silicon Metal
    else "cpu"
)
print(f"Using {device} device")
# -----------------------

# --- Dataset Class (from v0.0) ---
# Kept for potential direct use, though TensorDataset is often simpler
# class CustomDatasetTrain(Dataset):
#     def __init__(self, X_train, y_train):
#         self.features = X_train
#         self.label = y_train

#     def __len__(self):
#         return len(self.label)

#     def __getitem__(self, idx):
#         features = self.features[idx]
#         label = self.label[idx]
#         # Removed the dictionary wrapper for direct tuple return, common for PyTorch
#         return features, label

# --- Helper for LSTM Output (from v0.0) ---
class extract_tensor(nn.Module):
    def forward(self,x):
        # Input x is tuple (tensor, (hn, cn)) from LSTM
        tensor, (hn, cn) = x
        # Get the output features from the last time step
        # tensor shape: (batch, seq_len, num_directions * hidden_size) if batch_first=True
        return tensor[:, -1, :] # Return (batch, num_directions * hidden_size)

# --- Neural Network Definition (Updated Dimensions) ---
class NeuralNetwork(nn.Module):
    # Updated defaults: embedding_dim=32, hidden_dim=64
    def __init__(self, vocab_size=28, embedding_dim=32, hidden_dim=64, num_layers=1, bidirectional=True, dropout=0.2):
        super().__init__()
        # --- Parameters Updated based on request ---
        # vocab_size: Needs to match your character mapping size + pad/unknown (Assuming 28: a-z, _, pad=0)
        # embedding_dim: Set to 32.
        # hidden_dim: Set to 64.
        # num_layers: Set to 1 based on previous script, adjust if needed.
        # bidirectional: Set to True based on previous script, adjust if needed.
        # dropout: Added dropout within LSTM for regularization.
        # ----------------------------------------------------

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Layer sequence
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0) # Assuming 0 is padding index
        self.lstm = nn.LSTM(input_size=self.embedding_dim, # Use updated embedding_dim
                            hidden_size=self.hidden_dim,    # Use updated hidden_dim
                            num_layers=self.num_layers,
                            batch_first=True, # Important for data shape
                            dropout=dropout if num_layers > 1 else 0, # Dropout only between LSTM layers
                            bidirectional=bidirectional)
        self.extract_last = extract_tensor()
        # Linear layer input size is hidden_dim * num_directions (64 * 2 = 128)
        self.linear = nn.Linear(self.hidden_dim * self.num_directions, 26) # Output size 26 for a-z logits

    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x) # shape: (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded) # shape: (batch, seq_len, num_directions * hidden_dim)
        last_output = self.extract_last((lstm_out, _)) # shape: (batch, num_directions * hidden_dim)
        logits = self.linear(last_output) # shape: (batch, 26)
        return logits

# --- DataLoader Creation Function (modified from v0.0) ---
def create_dataloader(input_tensor, target_tensor, batch_size, shuffle=True):
    # Use standard TensorDataset
    dataset = TensorDataset(input_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# --- Save Model Function (modified from v0.0) ---
def save_model(model, filename="best_model_state.pt"):
    """Saves the model's state dictionary."""
    torch.save(model.state_dict(), filename)
    print(f"Model state saved to {filename}")

# --- Training Loop (from v1.0 snippet, device handling included) ---
def train_loop(data_loader, model, loss_fn, optimizer, epoch): # Added epoch back
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train() # Set model to training mode

    total_loss = 0
    epoch_start_time = time.time()
    batch_start_time = time.time()

    for batch, (X, y) in enumerate(data_loader):
        # Move data to the selected device
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred_logits = model(X)
        loss = loss_fn(pred_logits, y) # y should be float for BCEWithLogitsLoss

        # Backpropagation
        optimizer.zero_grad() # Zero gradients first
        loss.backward()       # Calculate gradients

        # --- Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # -----------------------

        optimizer.step()      # Update weights

        current_loss = loss.item() # Get float value of loss
        total_loss += current_loss

        # Print progress every 1000 batches (adjust interval if needed)
        if batch > 0 and batch % 1000 == 0:
            current_samples = (batch + 1) * len(X)
            time_elapsed = time.time() - batch_start_time
            batches_per_sec = 1000 / time_elapsed if time_elapsed > 0 else float('inf')
            print(f"  Epoch {epoch+1} Train Loss: {loss.item():>7f} [{current_samples:>7d}/{size:>7d}] Batch: {batch+1}/{num_batches} Speed: {batches_per_sec:.2f} b/s")
            batch_start_time = time.time() # Reset timer

    avg_train_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:>8f} (Time: {epoch_time:.2f}s)")
    return avg_train_loss # Return avg loss for potential tracking

# --- Validation Loop (from v1.0 snippet, device handling included) ---
def validation_loop(data_loader, model, loss_fn, epoch): # Added epoch back
    num_batches = len(data_loader)
    model.eval() # Set model to evaluation mode
    val_loss = 0
    total_samples = 0
    correct_predictions = 0
    total_possible_labels = 0

    with torch.no_grad(): # Disable gradient calculations
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred_logits = model(X)
            val_loss += loss_fn(pred_logits, y).item()

            # Calculate accuracy based on multi-label prediction
            pred_probs = torch.sigmoid(pred_logits) # Use sigmoid for multi-label probabilities
            pred_labels = (pred_probs > 0.5).float() # Threshold at 0.5
            total_samples += y.size(0) # Number of items in batch
            # Compare element-wise, sum correct labels across batch and labels
            correct_predictions += (pred_labels == y).sum().item()
            total_possible_labels += y.numel() # Total number of possible labels (batch_size * num_labels)

    val_loss /= num_batches
    # Accuracy is the proportion of correctly predicted labels (0 or 1) out of all possible labels
    accuracy = correct_predictions / total_possible_labels if total_possible_labels > 0 else 0
    print(f"Epoch {epoch+1} Validation Error: \n Approx. Label Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss # Return validation loss for early stopping & scheduler check


# --- train_model Function (Reconstructed & Enhanced, Updated Epochs/Patience) ---
# Updated defaults: epochs=20
def train_model(train_input, train_target, val_input, val_target, epochs=20, batch_size=128, learning_rate=1e-3):
    """
    Orchestrates the model training process.

    Args:
        train_input (Tensor): Training input features.
        train_target (Tensor): Training target labels.
        val_input (Tensor): Validation input features.
        val_target (Tensor): Validation target labels.
        epochs (int): Number of training epochs. (Default updated to 20)
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate for the optimizer.
    """
    print("--- Starting Model Training ---")
    print(f"Parameters: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}, EmbDim=32, HiddenDim=64") # Updated print

    # --- Create DataLoaders ---
    train_dataloader = create_dataloader(train_input, train_target, batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_input, val_target, batch_size, shuffle=False) # No shuffle for validation
    print(f"Train batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    # --- Initialize Model, Loss, Optimizer, Scheduler ---
    # NeuralNetwork uses updated dimensions: embedding_dim=32, hidden_dim=64
    model = NeuralNetwork().to(device) # Move model to device

    # Use BCEWithLogitsLoss for multi-label classification (predicting multiple letters)
    # Assumes target_tensor is float32 with 0s and 1s.
    loss_fn = nn.BCEWithLogitsLoss()

    # Use AdamW optimizer (often better than SGD for LSTMs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # weight_decay could be added here if needed

    # Learning Rate Scheduler: Reduce LR if validation loss plateaus
    # Patience here refers to LR scheduler patience, not early stopping
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # --- Training Loop with Validation and Saving Best Model ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 3 # Early stopping patience (updated to 3)

    best_model_state = None # Initialize best model state

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}\n-------------------------------") # Updated epoch print
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
        val_loss = validation_loop(val_dataloader, model, loss_fn, epoch)

        # Learning rate scheduler step based on validation loss
        scheduler.step(val_loss)

        # Check for improvement and save best model
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {val_loss:.6f}). Saving model...")
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save a copy of the best model state
            best_model_state = copy.deepcopy(model.state_dict())
            save_model(model, "best_model_state.pt") # Overwrite previous best
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s). Patience: {patience}")

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            # Load the best model state before finishing if it exists
            if best_model_state is not None:
                print("Loading best model state obtained during training...")
                model.load_state_dict(best_model_state)
            else:
                 print("Warning: Early stopping triggered, but no improvement was observed over initial state.")
            break

    print("--- Training Finished ---")
    if epochs_no_improve < patience:
         print("Completed all epochs.")
    # Optionally: Load the best saved model state back into the model object if needed elsewhere
    # Ensure the best model is loaded if training finished normally or stopped early with improvements
    if best_model_state is not None and epochs_no_improve >= patience : # Already loaded if stopped early
         pass # Already loaded the best state if stopped early
    elif best_model_state is not None: # If finished normally, load best state found
        print("Loading best model state found during training...")
        model.load_state_dict(best_model_state)


# Note: This script assumes it will be called via main.py,
# so it doesn't include an `if __name__ == '__main__':` block
# for direct execution, but you could add one for testing if needed.
# Example for testing:
# if __name__ == '__main__':
#     # Create dummy data matching expected shapes and types
#     # Input: LongTensor, shape (num_samples, sequence_length) e.g., (1000, 35)
#     # Target: FloatTensor, shape (num_samples, num_classes) e.g., (1000, 26)
#     seq_len = 35
#     num_classes = 26
#     vocab_size = 28 # 0-27
#     num_train_samples = 1024
#     num_val_samples = 256

#     # Dummy input data (integers between 0 and vocab_size-1)
#     dummy_train_input = torch.randint(0, vocab_size, (num_train_samples, seq_len), dtype=torch.long)
#     dummy_val_input = torch.randint(0, vocab_size, (num_val_samples, seq_len), dtype=torch.long)

#     # Dummy target data (multi-hot encoded, float type for BCEWithLogitsLoss)
#     dummy_train_target = torch.randint(0, 2, (num_train_samples, num_classes), dtype=torch.float)
#     dummy_val_target = torch.randint(0, 2, (num_val_samples, num_classes), dtype=torch.float)

#     print("Starting dummy training run...")
#     # Ensure tensors are on the correct device if needed (handled inside train_model)
#     train_model(dummy_train_input, dummy_train_target, dummy_val_input, dummy_val_target, epochs=3, batch_size=64) # Reduced epochs for quick test
#     print("Dummy training run finished.")