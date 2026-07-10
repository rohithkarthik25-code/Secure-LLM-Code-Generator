import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add current directory to path to allow importing dataset.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CodeDataset

class TokenEmbedder(nn.Module):
    """
    An Embedding module that maps discrete integer token IDs into dense continuous vector representations.
    This serves as the entry gate for input tokens before they are fed into a Transformer model.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        # PyTorch's nn.Embedding acts as a lookup table of shape (vocab_size, embedding_dim).
        # It holds 'vocab_size' number of vectors, where each vector has length 'embedding_dim'.
        # Initially, these vectors are randomized and will be learned/updated during backpropagation.
        self.embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token IDs with shape (batch_size, context_len)
        Returns:
            Tensor of embedding vectors with shape (batch_size, context_len, embedding_dim)
        """
        # Looking up the embedding vector for each token ID in the input tensor
        return self.embedding_table(x)

if __name__ == "__main__":
    # 1. Resolve paths and load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "code_dataset.txt")
    
    # Initialize the dataset with a context window of 32
    context_window = 32
    dataset = CodeDataset(data_path, context_len=context_window)
    
    # 2. Extract the vocabulary size from the tokenizer
    vocab_size = dataset.tokenizer.vocab_size
    
    # 3. Instantiate the embedding layer with embedding_dim = 64
    embedding_dim = 64
    embedder = TokenEmbedder(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # 4. Use a PyTorch DataLoader to create sample batches
    # We set batch_size = 4 (4 sequences processed in parallel)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Grab the first batch of (input, target) pairs from the iterator
    x_batch, y_batch = next(iter(dataloader))
    
    # 5. Pass the sample batch of token IDs through the embedding layer
    embedded_vectors = embedder(x_batch)
    
    # 6. Print the required information
    print("=" * 65)
    print(f"1. Vocabulary Size      : {vocab_size} unique characters")
    print(f"2. Input Batch Shape    : {list(x_batch.shape)}  -> [batch_size, context_len]")
    print(f"3. Output Batch Shape   : {list(embedded_vectors.shape)} -> [batch_size, context_len, embedding_dim]")
    print("=" * 65)
    
    # Print the first token's ID and its resulting 64-dimensional vector representation
    sample_batch_idx = 0
    sample_token_idx = 0
    
    token_id = x_batch[sample_batch_idx, sample_token_idx].item()
    char_representation = dataset.tokenizer.decode([token_id])
    token_vector = embedded_vectors[sample_batch_idx, sample_token_idx]
    
    print(f"Sample Token Details:")
    print(f"  Character             : {repr(char_representation)}")
    print(f"  Token ID (index)      : {token_id}")
    print(f"  Embedding Vector (64D):")
    # Pretty print the 64-dimensional float vector, showing first 5 and last 5 elements
    vector_list = token_vector.tolist()
    formatted_vector = f"[{', '.join(f'{val:.4f}' for val in vector_list[:5])}, ..., {', '.join(f'{val:.4f}' for val in vector_list[-5:])}]"
    print(f"    {formatted_vector}")
    print("=" * 65)
