import os
import sys
import torch
import torch.nn.functional as F

# Add current directory to path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizer import CharTokenizer
from mini_gpt import MiniGPT

def main():
    # 1. Resolve paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We load the dataset text to initialize the exact same tokenizer/vocab
    data_path = os.path.join(current_dir, "data", "code_dataset.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # 2. Instantiate the tokenizer
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    
    # 3. Instantiate the MiniGPT model with identical hyperparameters
    model = MiniGPT(vocab_size=vocab_size, embedding_dim=64, max_seq_len=256, num_heads=4)
    
    # Load the saved weights from mini_gpt_model.pth
    weights_path = os.path.join(current_dir, "mini_gpt_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained model weights not found at: {weights_path}. Please run train.py first.")
        
    print(f"Loading model weights from: {weights_path}")
    # Load model weights, mapping to CPU so it runs on any system
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    # 5. Put the model in evaluation mode
    # This disables layers like dropout or batchnorm running in train mode
    model.eval()
    
    # Sampling configurations for next-token generation
    # - temperature: lower values (e.g. 0.6) make output more confident/deterministic;
    #   higher values (e.g. 1.0) add variation/creativity.
    # - top_k: keeps only the top 'k' most probable characters, preventing the model
    #   from selecting very unlikely (garbage) characters.
    temperature = 0.8
    top_k = 5
    
    # Define a set of prompts to test the model
    prompts = [
        "def add(a, b):",
        "def is_even(n):",
        "for i in range(10):"
    ]
    
    print("=" * 65)
    print("Mini GPT Code Generation Demo (Temperature + Top-k Sampling)")
    print(f"Configuration: Temperature = {temperature}, Top-k = {top_k}")
    print("=" * 65)
    
    for prompt in prompts:
        print(f"Prompt: {repr(prompt)}")
        print("Generated Output:")
        print("-" * 45)
        
        # 4. Encode prompt into token IDs
        # input_ids shape: (batch_size=1, seq_len)
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        # Disable gradient calculations to speed up generation and save memory
        with torch.no_grad():
            # 9. Repeat generation for 100 characters
            for _ in range(100):
                # Context Cropping Protection:
                # If the sequence grows longer than our max supported length (256),
                # we slice it to keep only the last 256 tokens so positional embeddings don't crash.
                context_ids = input_ids[:, -model.max_seq_len:]
                
                # Get prediction logits from the model
                # logits shape: (batch_size=1, seq_len, vocab_size)
                logits, _ = model(context_ids)
                
                # Extract predictions for the VERY LAST character in the sequence
                # last_token_logits shape: (vocab_size,)
                last_token_logits = logits[0, -1, :]
                
                # 6. Apply temperature scaling
                scaled_logits = last_token_logits / temperature
                
                # 7. Apply top-k filtering
                # Keep only the top-k highest scoring tokens and set all other options to -inf.
                # This ensures the model never picks complete garbage syntax characters.
                v, ix = torch.topk(scaled_logits, min(top_k, vocab_size))
                filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                filtered_logits.scatter_(0, ix, v)
                
                # Turn the top-k logits into a probability distribution via softmax
                probs = F.softmax(filtered_logits, dim=-1)
                
                # Sample a character token ID from the probability distribution
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 8. Append the predicted character token ID to the input sequence
                # shape: (1, seq_len + 1)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long)], dim=1)
                
                # --- Stopping Heuristics ---
                # Decode the text generated so far and extract the suffix (characters appended after the prompt)
                current_text = tokenizer.decode(input_ids[0].tolist())
                generated_suffix = current_text[len(prompt):]
                
                # Heuristic 1: Stop if the model starts defining a new function
                if "def " in generated_suffix:
                    # Truncate at the start of the new "def " definition
                    cutoff_idx = len(prompt) + generated_suffix.index("def ")
                    input_ids = torch.tensor([tokenizer.encode(current_text[:cutoff_idx])], dtype=torch.long)
                    break
                    
                # Heuristic 2: Stop if the model outputs two consecutive newlines (separating blocks)
                if "\n\n" in generated_suffix:
                    # Truncate at the double newline
                    cutoff_idx = len(prompt) + generated_suffix.index("\n\n") + 2
                    input_ids = torch.tensor([tokenizer.encode(current_text[:cutoff_idx])], dtype=torch.long)
                    break

                
        # 10. Decode generated token IDs back into text
        generated_sequence = input_ids[0].tolist()
        decoded_text = tokenizer.decode(generated_sequence)
        
        # 11. Print the generated code
        print(decoded_text)
        print("=" * 65)


if __name__ == "__main__":
    main()
