import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import copy 
import time

# Import the custom class given for the assignment
from neural_network import NeuralNetwork 

# --- GLOBAL CONFIGURATION ---
# Detect if a GPU is available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'pendulum_dataset.npz')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'value_function_model.pt')


# --- DATA PREPARATION FUNCTION ---
def prepare_data(filepath, batch_size=64, train_split=0.8):
    """
    Loads dataset, normalizes data, splits into Train/Test, and creates PyTorch DataLoaders.
    
    Args:
    filepath: Path to the .npz dataset file.
    batch_size: Size of mini-batches for training/testing.
    train_split: Fraction of data to use for training (rest for testing).
    
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    print(f"--- Loading data from {filepath} ---")
    data = np.load(filepath)
    X = data['inputs']   # State [q, dq]
    Y = data['targets']  # Cost 

    # --- Normalization (Scaling) ---
    # We scale inputs to [-1, 1] range for better network performance
    X_max = np.max(np.abs(X), axis=0)
    X_scaled = X / X_max

    # We also scale the target Y to [-1, 1]
    Y_max = np.max(np.abs(Y))
    Y_scaled = (Y / Y_max).reshape(-1, 1) #Reshape Y to match the network output shape

    # --- Split Train/Test ---
    # 80% Train, 20% Test
    n_samples = X.shape[0]
    n_train = int(n_samples * train_split)
    
    # Generate random indices to shuffle the data before splitting
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Create NumPy arrays for Train and Test
    X_train_np, Y_train_np = X_scaled[train_idx], Y_scaled[train_idx]
    X_test_np, Y_test_np = X_scaled[test_idx], Y_scaled[test_idx]

    # --- Tensor Conversion ---
    # Convert NumPy arrays to PyTorch Tensors and move them to the Device (GPU/CPU)
    X_train = torch.from_numpy(X_train_np).float().to(DEVICE)
    Y_train = torch.from_numpy(Y_train_np).float().to(DEVICE)
    X_test = torch.from_numpy(X_test_np).float().to(DEVICE)
    Y_test = torch.from_numpy(Y_test_np).float().to(DEVICE)

    # --- DataLoaders ---
    # Create DataLoaders to handle batching
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    # Shuffle=True is important for training to break temporal correlations
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    # Shuffle=False for testing (order doesn't matter for evaluation)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
  
  
# --- TRAINING STEP FUNCTION ---
def train_one_epoch(model, dataloader, optimizer):
    """
    Performs one full pass (epoch) of training.
    (Epoch = all training samples processed once).
    The trainig is done in mini-batches provided by the dataloader.
    
    Args:
        model: The neural network model to train.
        dataloader: PyTorch DataLoader providing training batches.
        optimizer: The optimizer used for updating model weights.
        
    Returns:
        avg_loss: Average loss over the epoch.
    """
    model.train() # Set model to training mode (enables gradients)
    running_loss = 0.0
    
    # NB: dataloader provides batches of (inputs, targets):
    #       inputs:  Batch of states [q, dq] -> Shape: (Batch_Size, 2)
    #       targets: Batch of values       -> Shape: (Batch_Size, 1)
    for inputs, targets in dataloader:
        # Compute predicted outputs of the network for the current batch
        predictions = model(inputs)
        
        # Compute loss: Compare predictions with actual targets
        loss_batch = F.mse_loss(predictions, targets)
        
        # Backward pass & Optimization
        optimizer.zero_grad()   # Clear previous gradients for the current batch
        loss_batch.backward()   # Compute the gradients
        optimizer.step()        # Update model weights along the gradient direction
        
        #NB: The 2 lines above pereform the gradient descent step essentially 

        running_loss += loss_batch.item()
    
    # Return average loss for this epoch
    loss_epoch = running_loss / len(dataloader)
    return loss_epoch


# --- TESTING / EVALUATION FUNCTION ---
def test_network(model, dataloader):
    """
    Tests the network performance on the Test set.
    
    Args:
        model: The neural network model to test.
        dataloader: PyTorch DataLoader providing test batches.
        
    Returns:
        avg_loss: Average loss over the test set.
        
    This function body is similar to train_one_epoch, but without weight updates. 
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0

    # Use torch.no_grad() to disable gradient calculation (memory saving).
    with torch.no_grad():
        for inputs, targets in dataloader:
            predictions = model(inputs)
            
            # We calculate loss just to see how wrong we are
            loss_batch = F.mse_loss(predictions, targets)

            running_loss += loss_batch.item()

    # Return average loss
    avg_loss = running_loss / len(dataloader)
    return avg_loss


# --- LOW PASS FILTER ---
def low_pass_filter(data, alpha=0.1):
    """
    Applies an exponential moving average filter to smooth the data.
    """
    if not data: return []
    filtered_data = len(data) * [data[0]]
    for i in range(1, len(data)):
        filtered_data[i] = (1 - alpha) * filtered_data[i-1] + alpha * data[i]
    return filtered_data


# --- MAIN ---
def main():
    # Hyperparameters
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 2000
    FILTER_ALPHA = 0.1

    # Setup
    train_loader, test_loader = prepare_data(DATA_PATH, BATCH_SIZE)

    # Model, setup on DEVICE (GPU/CPU)
    model = NeuralNetwork(input_size=2, 
                          hidden_size=HIDDEN_SIZE, 
                          output_size=1, 
                          ub=1.0).to(DEVICE)
    
    # Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Starting Training Loop on {DEVICE} ---")
    
    # Tracking variables
    best_test_loss = float('inf')    
    best_model_wts = copy.deepcopy(model.state_dict())  # To store best model weights 
    
    # NB:deep copy needed because python uses references (pointers) that would get updated otherwise
    
    # History lists initialization
    train_history = []
    test_history = []

    # Just to measure time of training
    start_time = time.time()
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        
        # PHASE 1: TRAIN
        train_loss = train_one_epoch(model, train_loader, optimizer)
        
        # PHASE 2: VALIDATION 
        test_loss = test_network(model, test_loader)
        
        # Storage
        train_history.append(train_loss)
        test_history.append(test_loss)

        # Save Best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # Logging
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    
    # Training finished  
    end_time = time.time() 
    elapsed_time = end_time - start_time
    
    # Convert time to minutes and seconds
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)

    print(f"--- Finished in {mins}m {secs}s. Best Test Loss: {best_test_loss:.6f} ---")
    
    # Save best weights to file
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Apply Filter
    train_smooth = low_pass_filter(train_history, alpha=FILTER_ALPHA)
    test_smooth = low_pass_filter(test_history, alpha=FILTER_ALPHA)
    
    # --- PLOTTING  ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- PLOT 1: Raw Data (Noisy) ---
    ax1.set_title("1. Raw Loss History (Noisy)", fontsize=10)
    ax1.plot(train_history, alpha=0.6, color='blue', label='Train (Raw)')
    ax1.plot(test_history, alpha=0.6, color='orange', label='Test (Raw)')
    ax1.set_ylabel('MSE Loss (Log Scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # --- PLOT 2: Filtered Data (Smoothed) ---
    ax2.set_title(f"2. Smoothed History (Low-Pass Filter alpha={FILTER_ALPHA})", fontsize=10)
    ax2.plot(train_smooth, color='blue', linewidth=2, label='Train (Filtered)')
    ax2.plot(test_smooth, color='orange', linewidth=2, label='Test (Filtered)')
    ax2.set_xlabel('Epochs') 
    ax2.set_ylabel('MSE Loss (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()