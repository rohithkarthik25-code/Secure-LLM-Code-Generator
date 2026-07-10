import os

class CharTokenizer:
    """
    A simple character-level tokenizer for the mini code LLM.
    It maps each unique character in the training text to a unique integer index and vice versa.
    """
    def __init__(self, text: str):
        # 2. Extract all unique characters from the text and sort them to make the mapping deterministic
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 3. Create character-to-index mapping (stoi)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        
        # 4. Create index-to-character mapping (itos)
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, text: str) -> list:
        """
        Encodes a string of text into a list of integer tokens.
        """
        # Iterate over characters and lookup their corresponding index
        return [self.stoi[c] for c in text]
        
    def decode(self, tokens: list) -> str:
        """
        Decodes a list of integer tokens back into a string of text.
        """
        # Iterate over tokens and lookup their corresponding character
        return "".join([self.itos[i] for i in tokens])

if __name__ == "__main__":
    # 1. Read data/code_dataset.txt relative to this script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "data", "code_dataset.txt")
    
    print(f"Reading dataset from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Initialize the tokenizer with the corpus text to build the vocabulary
    tokenizer = CharTokenizer(text)
    
    # 7. Print vocabulary size
    print("Dataset Size:", len(text))
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters.")
    print("Vocabulary characters:", repr("".join(tokenizer.chars)))
    print("-" * 50)
    
    # 5. Encode the entire dataset into integer tokens
    encoded_tokens = tokenizer.encode(text)
    
    # 8. Print the first 100 encoded tokens
    print("First 100 encoded tokens:")
    print(encoded_tokens[:100])
    print("-" * 50)
    
    # 6. Decode the tokens back into text
    decoded_text = tokenizer.decode(encoded_tokens)
    
    # 9. Verify that decoding reconstructs the original text exactly
    verification_success = (text == decoded_text)
    print("Verification of Decoding:")
    print(f"Decoded text matches original exactly: {verification_success}")
    
    # Print a small sample of decoded text to visually verify
    print("-" * 50)
    print("Sample decoded text (first 120 chars):")
    print(decoded_text[:120])
