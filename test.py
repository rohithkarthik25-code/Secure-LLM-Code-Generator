import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"C:\Users\L509\gemma-4-e4b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map="cpu"
)

messages = [
    {
        "role": "user",
        "content": "Write a Python function to reverse a string."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")

print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated,
    skip_special_tokens=True
)

print("\n========== RESPONSE ==========\n")
print(response)