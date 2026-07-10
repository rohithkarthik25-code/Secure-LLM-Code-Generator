import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the current directory to Python's system path to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CodeDataset
from mini_gpt import MiniGPT

def train():
    # Set the device: Use GPU (cuda) if available for faster training, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Resolve paths and load the custom PyTorch dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "code_dataset.txt")
    
    # Using context window (seq_len) of 128 characters
    context_window = 128
    dataset = CodeDataset(data_path, context_len=context_window)
    vocab_size = dataset.tokenizer.vocab_size
    
    # 2. Create the PyTorch DataLoader with batch_size=4
    # We shuffle the dataset so the model doesn't memorize the sequential order of samples
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Load the MiniGPT model and move its parameters to the chosen device
    model = MiniGPT(vocab_size=vocab_size, embedding_dim=64, max_seq_len=256, num_heads=4)
    model = model.to(device)
    
    # 9. Print the total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 55)
    print(f"Training MiniGPT Model:")
    print(f"  Vocab Size     : {vocab_size} characters")
    print(f"  Total Parameters: {total_params} parameters")
    print("=" * 55)
    
    # 4. Initialize the AdamW optimizer with learning_rate = 1e-3 (0.001)
    # AdamW is an improved version of Adam with weight decay regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Set model to training mode (enables dropout if present, updates batch norm parameters, etc.)
    model.train()
    
    # 5. Train for 50 epochs
    num_epochs = 50

    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for x, y in dataloader:
            # Move batch inputs (x) and targets (y) to the target device
            x, y = x.to(device), y.to(device)
            
            # 6 & 8. Training Steps:
            
            # --- Stage A: Forward Pass & Loss Computation ---
            # We pass both input sequences 'x' and target labels 'y' into our model.
            # The model automatically runs them through embeddings, transformer blocks,
            # and computes Cross-Entropy Loss internally by comparing predictions (logits) to targets (y).
            logits, loss = model(x, targets=y)
            
            # --- Stage B: Zero Gradients ---
            # PyTorch accumulates gradients by default. We must clear the gradients
            # from the previous step before calculating gradients for this batch.
            optimizer.zero_grad()
            
            # --- Stage C: Backpropagation (Backward Pass) ---
            # Computes the gradient of the loss with respect to all learnable model parameters.
            # It traverses backward through the computation graph, applying the chain rule.
            loss.backward()
            
            # --- Stage D: Optimizer Step ---
            # Updates the weights and biases of the model using the calculated gradients
            # and the AdamW update rule (taking a step in the direction that minimizes loss).
            optimizer.step()
            
            # Accumulate loss metrics
            epoch_loss += loss.item()
            num_batches += 1
            
        # 7. Print the average loss for this epoch
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch + 1:02d}/{num_epochs:02d} | Average Loss: {avg_loss:.4f}")
        
    print("=" * 55)
    print("Training Complete!")
    print("=" * 55)
    
    # Save the trained model parameters (state dict) to the project folder
    save_path = os.path.join(current_dir, "mini_gpt_model.pth")
    print(f"Saving trained model weights to: {save_path}")
    torch.save(model.state_dict(), save_path)
    print("Save complete!")


if __name__ == "__main__":
    train()
