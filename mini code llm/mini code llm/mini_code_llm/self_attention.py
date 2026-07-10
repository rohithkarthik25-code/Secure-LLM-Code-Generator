import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    A single-head causal self-attention layer.
    This module allows tokens in a sequence to communicate with each other, focusing
    on relevant past tokens while ignoring future tokens (due to causal masking).
    """
    def __init__(self, embedding_dim: int = 64, head_size: int = 64, max_seq_len: int = 256):
        super().__init__()
        
        # 2. Key, Query, and Value projections.
        # These linear layers project the input embeddings into three separate spaces:
        # - Queries (Q): "What am I looking for?"
        # - Keys (K): "What information do I contain?"
        # - Values (V): "If you attend to me, what information do I offer?"
        # We set bias=False as is common in attention layers.
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
        # Causal Mask: Lower triangular matrix of ones.
        # This is registered as a 'buffer' so PyTorch knows it is a state variable of the model,
        # but not a learnable parameter that requires gradients.
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            out: Attention output tensor of shape (batch_size, seq_len, head_size)
            weights: Attention weights matrix of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # Project input embeddings into Queries, Keys, and Values
        # Shapes: (B, T, C) -> (B, T, head_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 3. Compute attention scores using scaled dot-product attention.
        # We compute raw affinities by multiplying Query and Key matrices:
        # (batch_size, seq_len, head_size) @ (batch_size, head_size, seq_len) -> (batch_size, seq_len, seq_len)
        # We scale by dividing by the square root of the head size (d_k) to prevent gradients from vanishing.
        d_k = k.shape[-1]
        scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Causal Masking: For autoregressive decoding (predicting next token), a token at index t
        # must not see tokens at t+1, t+2, etc. We set future positions to negative infinity (-inf)
        # so that softmax will assign them a probability weight of exactly 0.
        masked_scores = scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        
        # 4. Apply softmax to obtain attention weights (probabilities summing to 1 across columns).
        # Shape remains: (batch_size, seq_len, seq_len)
        weights = F.softmax(masked_scores, dim=-1)
        
        # 5. Produce attention output vectors by taking the weighted sum of Value vectors:
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, head_size) -> (batch_size, seq_len, head_size)
        out = weights @ v
        
        return out, weights

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # 10. Assume embedding dimension = 64
    embedding_dim = 64
    head_size = 64
    
    # Instantiate the self-attention module
    attention_layer = SelfAttention(embedding_dim=embedding_dim, head_size=head_size)
    
    # Define a sample batch
    # Shape: (batch_size=4, seq_len=8, embedding_dim=64)
    # Using seq_len=8 instead of 32 for easier visualization of the attention matrix.
    batch_size = 4
    seq_len = 8
    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Pass through the attention layer
    output_vectors, attention_weights = attention_layer(dummy_input)
    
    # 6 & 7. Print the required shapes
    print("=" * 60)
    print(f"Input Embedding Shape       : {list(dummy_input.shape)} -> [B, T, C]")
    print(f"Output Attention Shape      : {list(output_vectors.shape)} -> [B, T, head_size]")
    print(f"Attention Weights Shape     : {list(attention_weights.shape)} -> [B, T, T]")
    print("=" * 60)
    
    # Print the attention weight matrix for the first batch sample
    # to show the causal lower-triangular pattern
    print("Attention Weights Matrix (Sample 1):")
    sample_weights = attention_weights[0] # Shape (seq_len, seq_len)
    
    for row in range(seq_len):
        row_str = " ".join(f"{val:.3f}" for val in sample_weights[row].tolist())
        print(f"  Token {row + 1} attends to: [{row_str}]")
    print("=" * 60)
    print("Notice that the upper-triangular values are 0.000 (causal masking).")
