import os
import sys
import torch
from torch.utils.data import Dataset

# Add the directory containing tokenizer.py to the system path so we can import it easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizer import CharTokenizer

class CodeDataset(Dataset):
    """
    A PyTorch Dataset class that prepares tokenized text data for autoregressive language model training.
    For each sample, it provides an input sequence of length context_len and a target sequence of the
    same length, shifted by 1 position (predicting the next token).
    """
    def __init__(self, data_path: str, context_len: int = 128):
        # 1. Check if dataset file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
        # 2. Read the raw text data
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # 3. Initialize tokenizer and encode the corpus into integer token IDs
        self.tokenizer = CharTokenizer(text)
        self.tokens = self.tokenizer.encode(text)
        
        # 4. Convert python list of token IDs into a PyTorch tensor
        # We use torch.long (int64) since token IDs will be used as indices in embedding layers
        self.data_tensor = torch.tensor(self.tokens, dtype=torch.long)
        
        # 5. Set the context window size
        self.context_len = context_len
        
    def __len__(self) -> int:
        # The total number of valid samples we can extract from the dataset.
        # Since each sample requires context_len tokens for input and 1 additional token for target,
        # we subtract context_len from the total token length.
        return len(self.data_tensor) - self.context_len
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 6. Extract input (x) and target (y) token sequences
        # Input sequence (x): slice from index idx to idx + context_len
        x = self.data_tensor[idx : idx + self.context_len]
        
        # Target sequence (y): slice from index idx + 1 to idx + context_len + 1
        # This shifts the sequence by 1 position to the right (predicting the next token)
        y = self.data_tensor[idx + 1 : idx + self.context_len + 1]
        
        return x, y

if __name__ == "__main__":
    # Resolve the data file path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "code_dataset.txt")
    
    # 7. Create dataset instance with a context window of 128
    context_window = 128
    dataset = CodeDataset(data_path, context_len=context_window)

    
    print(f"Loaded dataset from: {data_path}")
    print(f"Total tokens in corpus: {len(dataset.data_tensor)}")
    print(f"Total training samples: {len(dataset)}")
    print("=" * 60)
    
    # 8. Print first 3 sample input-target pairs for inspection
    print("Printing first 3 training samples:\n")
    for i in range(3):
        x, y = dataset[i]
        print(f"--- Sample {i + 1} ---")
        print(f"Input token IDs (x) :\n  {x.tolist()}")
        print(f"Target token IDs (y):\n  {y.tolist()}")
        print()
        
        # Decode the IDs back to text characters to visualize the shift
        x_text = dataset.tokenizer.decode(x.tolist())
        y_text = dataset.tokenizer.decode(y.tolist())
        print(f"Input text (x) :\n  {repr(x_text)}")
        print(f"Target text (y):\n  {repr(y_text)}")
        print("-" * 60)
