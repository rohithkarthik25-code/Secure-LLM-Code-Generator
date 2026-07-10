import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A Feed-Forward Network (FFN) block.
    In a Transformer, the Multi-Head Attention layer allows tokens to communicate/exchange information.
    Directly after attention, this FFN layer operates on each token position independently and in parallel.
    It acts as a small Multi-Layer Perceptron (MLP) that refines and transforms the representations of tokens individually.
    """
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        # 3, 4 & 5. Define the sequential neural network layers.
        # - We expand the dimensions by 4x (64 -> 256) to give the network extra parameter capacity to learn.
        # - We apply a non-linear activation function (ReLU) to allow the network to model non-linear patterns.
        # - We project the representation back down to the original embedding dimension (256 -> 64).
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), # Linear layer 1
            nn.ReLU(),                                   # Activation function
            nn.Linear(4 * embedding_dim, embedding_dim)  # Linear layer 2 (Projection layer)
        )
        
    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            verbose: If True, prints shape transformations
        Returns:
            Projected and activated output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Step 1: Expand sequence representation using the first Linear layer (64 -> 256)
        x1 = self.net[0](x)
        if verbose:
            print(f"  After Linear 1 (expansion): {list(x1.shape)}")
        
        # Step 2: Apply the ReLU activation function
        x2 = self.net[1](x1)
        if verbose:
            print(f"  After ReLU (activation)   : {list(x2.shape)}")
        
        # Step 3: Project back to the original embedding dimension (256 -> 64)
        x3 = self.net[2](x2)
        if verbose:
            print(f"  After Linear 2 (projection): {list(x3.shape)}")
        
        return x3


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # 2. Input dimension = 64
    embedding_dim = 64
    
    # Instantiate the Feed-Forward module
    ff_layer = FeedForward(embedding_dim=embedding_dim)
    
    # 10. Create a small test example using random input of shape [4, 32, 64]
    batch_size = 4
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    
    # 7. Print the layer architecture
    print("=" * 65)
    print("Feed-Forward Layer Architecture:")
    print(ff_layer)
    print("=" * 65)
    
    # Print intermediate transformations
    print("Intermediate Layer Transformations:")
    # Pass input through the layer with verbose=True to print shape transformations
    output_vectors = ff_layer(dummy_input, verbose=True)

    print("=" * 65)
    
    # 6. Print input shape and output shape
    print(f"Input Embedding Shape   : {list(dummy_input.shape)} -> [B, T, embedding_dim]")
    print(f"Output Embedding Shape  : {list(output_vectors.shape)} -> [B, T, embedding_dim]")
    
    # Print the learnable parameters count
    print(f"Parameter Count         : {sum(p.numel() for p in ff_layer.parameters())} parameters")
    print("=" * 65)
    
    # Print sample output vector (following our learning pattern)
    print("Sample Output Vector (first 5 values of first token in batch 1):")
    print(output_vectors[0, 0, :5])
    print("=" * 65)

