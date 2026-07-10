import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add current directory to path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock
from tokenizer import CharTokenizer

class MiniGPT(nn.Module):
    """
    A complete Decoder-only GPT (Generative Pre-trained Transformer) model built from scratch.
    It takes sequence token IDs, generates token and positional embeddings, passes them
    through a stack of Transformer Decoder Blocks, and projects them to vocabulary logits.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64, max_seq_len: int = 256, num_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 2. Token embedding lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        
        # 3. Positional embedding layer
        self.position_embedding_layer = PositionalEmbedding(max_seq_len=max_seq_len, embedding_dim=embedding_dim)
        
        # 4. Stack of 2 Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, max_seq_len=max_seq_len)
            for _ in range(2)
        ])
        
        # 5. Final layer normalization (stabilizes logits before projection)
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # 6. Language modeling head (projects 64D vectors back to character probability logits)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Tensor of input token IDs of shape (batch_size, seq_len)
            targets: Optional tensor of target token IDs of shape (batch_size, seq_len)
            verbose: If True, prints shapes after every stage of execution
        Returns:
            logits: Prediction scores for every character in the vocabulary of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets are provided) otherwise None
        """
        batch_size, seq_len = idx.shape
        
        if verbose:
            print(f"  [Stage 0] Input Token IDs Shape            : {list(idx.shape)} -> [B, T]")
            
        # 2. Stage 1: Token Embeddings
        # Shape: (B, T) -> (B, T, C)
        tok_emb = self.token_embedding_table(idx)
        if verbose:
            print(f"  [Stage 1] Token Embeddings Shape           : {list(tok_emb.shape)} -> [B, T, C]")
            
        # 3. Stage 2: Positional Embeddings
        # Shape: (T, C)
        pos_emb = self.position_embedding_layer(tok_emb)
        if verbose:
            print(f"  [Stage 2] Positional Embeddings Shape      : {list(pos_emb.shape)} -> [T, C]")
            
        # Stage 3: Combine Token & Positional Embeddings via broadcasting addition
        # Shape: (B, T, C) + (T, C) -> (B, T, C)
        x = tok_emb + pos_emb
        if verbose:
            print(f"  [Stage 3] Combined (Token + Pos) Shape     : {list(x.shape)} -> [B, T, C]")
            
        for i, block in enumerate(self.blocks):
            x, _ = block(x, verbose=verbose)
            if verbose:
                print(f"  [Stage 4] After Transformer Block {i + 1} Shape: {list(x.shape)} -> [B, T, C]")

                
        # 5. Stage 5: Final Layer Normalization
        # Shape remains: (B, T, C)
        x = self.ln_f(x)
        if verbose:
            print(f"  [Stage 5] After Final LayerNorm Shape      : {list(x.shape)} -> [B, T, C]")
            
        # 6. Stage 6: Language Modeling Head (Logits)
        # Shape: (B, T, C) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        if verbose:
            print(f"  [Stage 6] Output Logits Shape              : {list(logits.shape)} -> [B, T, vocab_size]")
            
        # Optional: Compute loss if targets are provided (useful for training loop later)
        loss = None
        if targets is not None:
            # logits: (B, T, vocab_size) -> reshape to (B * T, vocab_size)
            # targets: (B, T) -> reshape to (B * T)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Load tokenizer vocab size from dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "data", "code_dataset.txt")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    
    # Instantiate the MiniGPT model
    embedding_dim = 64
    max_seq_len = 256
    num_heads = 4
    model = MiniGPT(vocab_size=vocab_size, embedding_dim=embedding_dim, max_seq_len=max_seq_len, num_heads=num_heads)
    
    # 10. Print the total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"Mini GPT Model Configuration:")
    print(f"  Vocab Size     : {vocab_size} characters")
    print(f"  Embedding Dim  : {embedding_dim}")
    print(f"  Stack Depth    : 2 Blocks")
    print(f"  Parameter Count: {total_params} parameters")
    print("=" * 70)
    
    # 7 & 9. Test forward pass with verbose shape printing
    batch_size = 4
    seq_len = 32
    # Create dummy token IDs of shape [batch_size, seq_len]
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Step-by-Step Model Execution Shapes:")
    print("=" * 70)
    logits, _ = model(dummy_input_ids, verbose=True)
    print("=" * 70)
    
    # 8. Verify the final shape of the output logits
    print(f"Final Logits Shape: {list(logits.shape)} -> [B, T, vocab_size]")
    print("=" * 70)
