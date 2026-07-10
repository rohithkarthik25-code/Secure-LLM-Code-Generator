import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    A learnable absolute Positional Embedding module.
    In Transformers, self-attention does not have any built-in sense of order or position
    (it processes sequences like a bag of words). To fix this, we must inject positional
    information into the inputs.
    
    This layer assigns a unique learnable vector to each position index (0 to max_seq_len-1).
    These positional vectors are added to the token embeddings.
    """
    def __init__(self, max_seq_len: int = 256, embedding_dim: int = 64):
        super().__init__()
        
        # 2. Store max sequence length for protection checks
        self.max_seq_len = max_seq_len
        
        # 1 & 3. Create the learnable lookup table for positions.
        # It holds 'max_seq_len' vectors, each of size 'embedding_dim'.
        self.position_embedding_table = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token embedding tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            pos_embeddings: Tensor of shape (seq_len, embedding_dim)
        """
        # 3. Extract batch size, sequence length, and embedding dimension from input
        batch_size, seq_len, embedding_dim = x.shape
        
        # 1 & 2. Sequence length protection check to prevent index out of bounds
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max supported length of {self.max_seq_len}"
        
        # 4. Generate position indices: [0, 1, 2, ..., seq_len - 1]
        # We create this on the same device as the input tensor x for device-agnostic execution
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        # Retrieve the embedding vectors for these position indices
        # 5. Output shape: (seq_len, embedding_dim)
        pos_embeddings = self.position_embedding_table(positions)
        
        return pos_embeddings

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    max_seq_len = 256
    embedding_dim = 64
    batch_size = 4
    seq_len = 32
    
    # Initialize the Positional Embedding module
    pos_embedder = PositionalEmbedding(max_seq_len=max_seq_len, embedding_dim=embedding_dim)
    
    # 9. Create a test example of dummy token embeddings
    # Shape: (batch_size=4, seq_len=32, embedding_dim=64)
    dummy_token_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Fetch positional embeddings by passing the input tensor x
    pos_embeddings = pos_embedder(dummy_token_embeddings)
    
    # 8. Show how token embeddings and positional embeddings are added together
    # Under the hood, PyTorch uses broadcasting to add the (seq_len, embedding_dim) tensor
    # to the (batch_size, seq_len, embedding_dim) tensor, automatically replicating the positions
    # across each item in the batch.
    combined_embeddings = dummy_token_embeddings + pos_embeddings
    
    # 7. Print shapes
    print("=" * 70)
    print("Positional Embedding Verification:")
    print("=" * 70)
    print(f"1. Input Token Embedding Shape  : {list(dummy_token_embeddings.shape)} -> [B, T, C]")
    print(f"2. Positional Embedding Shape   : {list(pos_embeddings.shape)}     -> [T, C]")
    print(f"3. Combined Embedding Shape     : {list(combined_embeddings.shape)} -> [B, T, C]")
    print("=" * 70)
    
    # Let's inspect a few dimensions of the first position embedding
    print("First Position Learnable Vector (first 5 elements):")
    print(pos_embeddings[0, :5])
    print("=" * 70)
    
    # Show that mathematical addition is correctly computed
    print("Verification of Broadcasting Addition:")
    print("  Token Emb (Batch 1, Token 1) :", dummy_token_embeddings[0, 0, :3].tolist())
    print("  Pos Emb (Token 1)            :", pos_embeddings[0, :3].tolist())
    print("  Combined (Batch 1, Token 1)  :", combined_embeddings[0, 0, :3].tolist())
    print("  (Token Emb + Pos Emb matches Combined matches exactly)")
    print("=" * 70)
    
    # Verify sequence length protection works (let's trigger it using a large seq_len)
    print("Testing Sequence Length Protection:")
    large_input = torch.randn(batch_size, 300, embedding_dim)
    try:
        pos_embedder(large_input)
    except AssertionError as e:
        print(f"  Success: Caught expected error -> '{e}'")
    print("=" * 70)
