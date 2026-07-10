import os
import sys
import torch
import torch.nn as nn

# Add current directory to path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):
    """
    A single Post-LayerNorm Transformer Decoder Block.
    This block integrates:
    1. Multi-Head Self-Attention (communication across tokens)
    2. Residual Skip Connection (helps gradients flow, preventing vanishing/exploding gradients)
    3. Layer Normalization (stabilizes activation values during training)
    4. Position-wise Feed-Forward Network (token-level refinement computation)
    """
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4, max_seq_len: int = 256):
        super().__init__()
        
        # 1 & 2. Instantiate custom MultiHeadAttention and FeedForward modules
        self.mha = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, max_seq_len=max_seq_len)
        self.ff = FeedForward(embedding_dim=embedding_dim)
        
        # 3. Instantiate two LayerNorm layers (one for attention, one for feed-forward)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor, verbose: bool = False) -> tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            verbose: If True, prints intermediate shapes and stages for learning
        Returns:
            out: Output tensor of shape (batch_size, seq_len, embedding_dim)
            attn_weights: List of attention weight tensors from the attention heads
        """
        if verbose:
            print(f"  [Stage 0] Input Shape                      : {list(x.shape)}")
            
        # Stage 1: Pass through Multi-Head Attention
        # Recall that our custom MultiHeadAttention returns (output_vectors, attention_weights)
        att_out, attn_weights = self.mha(x)
        if verbose:
            print(f"  [Stage 1] Multi-Head Attention Output Shape : {list(att_out.shape)}")
            
        # Stage 2: Residual Add + LayerNorm 1
        # We add the original input 'x' (residual link) to the attention output, and normalize
        x = self.ln1(x + att_out)
        if verbose:
            print(f"  [Stage 2] After Residual + LayerNorm 1 Shape: {list(x.shape)}")
            
        # Stage 3: Pass through Feed-Forward Network
        ff_out = self.ff(x, verbose=verbose)

        if verbose:
            print(f"  [Stage 3] Feed-Forward Output Shape        : {list(ff_out.shape)}")
            
        # Stage 4: Residual Add + LayerNorm 2
        # We add the pre-FFN state (residual link) to the FFN output, and normalize
        x = self.ln2(x + ff_out)
        if verbose:
            print(f"  [Stage 4] After Residual + LayerNorm 2 Shape: {list(x.shape)}")
            
        return x, attn_weights

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # 6. Use embedding_dim = 64 and num_heads = 4
    embedding_dim = 64
    num_heads = 4
    
    # Instantiate the Transformer Block
    block = TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads)
    
    # Create a dummy batch of embedding vectors
    batch_size = 4
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Run forward pass with verbose printing to explain each stage
    print("=" * 70)
    print("Transformer Block Step-by-Step Execution:")
    print("=" * 70)
    output_vectors, attn_weights = block(dummy_input, verbose=True)
    print("=" * 70)
    
    # 7. Print final input and output shapes
    print(f"Final Summary:")
    print(f"  Input Shape  : {list(dummy_input.shape)} -> [B, T, embedding_dim]")
    print(f"  Output Shape : {list(output_vectors.shape)} -> [B, T, embedding_dim]")
    print(f"  Block Learnable Parameters: {sum(p.numel() for p in block.parameters())} params")
    print("=" * 70)
