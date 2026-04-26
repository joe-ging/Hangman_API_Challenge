# train.py (Combined v1.0 - Merged, Corrected, and Reconstructed)

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
class CustomDatasetTrain(Dataset):
    # Kept for potential direct use, though TensorDataset is often simpler
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        # Removed the dictionary wrapper for direct tuple return, common for PyTorch
        return features, label

# --- Helper for LSTM Output (from v0.0) ---
class extract_tensor(nn.Module):
    def forward(self,x):
        # Input x is tuple (tensor, (hn, cn)) from LSTM
        tensor, (hn, cn) = x
        # Get the output features from the last time step
        # tensor shape: (batch, seq_len, num_directions * hidden_size) if batch_first=True
        return tensor[:, -1, :] # Return (batch, num_directions * hidden_size)

# --- Neural Network Definition (from v0.0, Corrected based on checkpoint error) ---
class NeuralNetwork(nn.Module):
    """
    神经网络架构解析:
    1. Embedding 层: 将字母索引映射为 64 维向量 (vocab_size -> embedding_dim)
    2. Bi-LSTM 层: 提取序列特征。hidden_dim=32，双向(bidirectional)意味着最终输出 32*2 = 64 维。
    3. Linear 层: 将 LSTM 的 64 维隐藏状态映射到 26 个字母的预测概率上。
    
    --- TensorFlow/Keras 等效代码 (面试展示用) ---
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
    #     tf.keras.layers.Dense(26)
    # ])
    """
    def __init__(self, vocab_size=28, embedding_dim=64, hidden_dim=32, num_layers=1, bidirectional=True, dropout=0.2):
        super().__init__()
        # --- Parameters Updated based on previous analysis ---
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Layer sequence
        # nn.Embedding: 嵌入层，将离散的字母索引变成 64 维的特征向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
        
        # nn.LSTM: 循环神经网络，负责记忆单词的结构
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0, 
                            bidirectional=bidirectional)
        
        self.extract_last = extract_tensor()
        
        # nn.Linear: 全连接层。输入维度 = hidden_dim (32) * num_directions (2) = 64
        # 输出 26 代表对全英文字母的打分
        self.linear = nn.Linear(hidden_dim * self.num_directions, 26) 

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


# --- train_model Function (Reconstructed & Enhanced based on v0.0 and best practices) ---
def train_model(train_input, train_target, val_input, val_target, epochs=15, batch_size=128, learning_rate=1e-3):
    """
    Orchestrates the model training process.

    Args:
        train_input (Tensor): Training input features.
        train_target (Tensor): Training target labels.
        val_input (Tensor): Validation input features.
        val_target (Tensor): Validation target labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate for the optimizer.
    """
    print("--- Starting Model Training ---")
    print(f"Parameters: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}")

    # --- Create DataLoaders ---
    train_dataloader = create_dataloader(train_input, train_target, batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_input, val_target, batch_size, shuffle=False) # No shuffle for validation
    print(f"Train batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    # --- Initialize Model, Loss, Optimizer, Scheduler ---
    # Ensure NeuralNetwork uses the corrected parameters matching the checkpoint if fine-tuning,
    # or the desired architecture if training from scratch.
    # Assuming corrected parameters from class definition above.
    model = NeuralNetwork().to(device) # Move model to device

    # Use BCEWithLogitsLoss for multi-label classification (predicting multiple letters)
    # Assumes target_tensor is float32 with 0s and 1s.
    loss_fn = nn.BCEWithLogitsLoss()

    # Use AdamW optimizer (often better than SGD for LSTMs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler: Reduce LR if validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # --- Training Loop with Validation and Saving Best Model ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 4 # Number of epochs to wait for improvement before stopping early

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
        val_loss = validation_loop(val_dataloader, model, loss_fn, epoch)

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save a copy of the best model state
            best_model_state = copy.deepcopy(model.state_dict())
            save_model(model, "best_model_state.pt") # Overwrite previous best
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            # Load the best model state before finishing
            print("Loading best model state...")
            model.load_state_dict(best_model_state)
            break

    print("Training Done!")
    # Optionally: Load the best saved model state back into the model object if needed elsewhere
    # model.load_state_dict(torch.load("best_model_state.pt", map_location=device))

# Note: This script assumes it will be called via main.py,
# so it doesn't include an `if __name__ == '__main__':` block
# for direct execution, but you could add one for testing if needed.