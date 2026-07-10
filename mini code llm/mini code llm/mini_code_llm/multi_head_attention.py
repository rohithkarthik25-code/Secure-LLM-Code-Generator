import os
import sys
import torch
import torch.nn as nn

# Add current directory to path to import SelfAttention from self_attention.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from self_attention import SelfAttention

class MultiHeadAttention(nn.Module):
    """
    A Multi-Head Attention layer.
    This module instantiates multiple independent SelfAttention heads, runs them in parallel,
    concatenates their outputs, and projects the result back to the embedding space.
    Running multiple heads allows the model to attend to different parts of the context simultaneously.
    """
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4, max_seq_len: int = 256):
        super().__init__()
        
        # 2. Calculate the size of each individual head.
        # Since we concatenate the outputs, we want num_heads * head_size to equal embedding_dim
        # So for embedding_dim=64 and num_heads=4, head_size = 16.
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.head_size = embedding_dim // num_heads
        
        # 1 & 3. Create 'num_heads' independent SelfAttention modules.
        # We use nn.ModuleList to register them properly as sub-modules of this layer.
        self.heads = nn.ModuleList([
            SelfAttention(embedding_dim=embedding_dim, head_size=self.head_size, max_seq_len=max_seq_len)
            for _ in range(num_heads)
        ])
        
        # 5. Final linear projection layer.
        # This mixes the concatenated outputs from all heads back together.
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            out: Projected output tensor of shape (batch_size, seq_len, embedding_dim)
            attention_weights: List of attention weight tensors from each head, each of shape (batch_size, seq_len, seq_len)
        """
        # 3. Run all attention heads in parallel.
        # We retrieve the attention output and attention weights for each head.
        head_results = [head(x) for head in self.heads]
        head_outputs = [result[0] for result in head_results]
        attention_weights = [result[1] for result in head_results]
        
        # 4. Concatenate head outputs along the last dimension (channels/features)
        # Shape: List of num_heads tensors of shape (B, T, head_size) -> (B, T, num_heads * head_size)
        # For our parameters, this is: [B, T, 16] * 4 -> [B, T, 64]
        out = torch.cat(head_outputs, dim=-1)
        
        # 5. Apply the final linear projection layer
        # Shape remains: (batch_size, seq_len, embedding_dim)
        out = self.proj(out)
        
        return out, attention_weights

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters (as per embedding layer output)
    embedding_dim = 64
    num_heads = 4
    
    # Instantiate the Multi-Head Attention module
    mha_layer = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
    
    # Create a dummy batch of embedding vectors
    # Shape: (batch_size=4, seq_len=32, embedding_dim=64)
    batch_size = 4
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Pass through the Multi-Head Attention layer
    output_vectors, attention_weights = mha_layer(dummy_input)
    
    # 6 & 7. Print the shapes
    print("=" * 65)
    print(f"Input Embedding Shape       : {list(dummy_input.shape)} -> [B, T, embedding_dim]")
    print(f"Individual Head Size        : {mha_layer.head_size} dimensions")
    print(f"Shape of each Head Output   : {[batch_size, seq_len, mha_layer.head_size]} -> [B, T, head_size]")
    print(f"Shape of Head 1 Attn Weights: {list(attention_weights[0].shape)} -> [B, T, T]")
    print(f"Output Attention Shape      : {list(output_vectors.shape)} -> [B, T, embedding_dim]")
    print("=" * 65)
    
    # Print a sample output vector and first head's weight matrix shape
    print("Sample Output Vector (first 5 values of first token in batch 1):")
    print(output_vectors[0, 0, :5])
    print("=" * 65)

