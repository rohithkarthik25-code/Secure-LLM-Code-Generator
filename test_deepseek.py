import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"C:\Users\L509\Desktop\llm project\DeepSeek-Coder-V2-Lite-Instruct"

print("=" * 70)
print("Loading DeepSeek-Coder-V2-Lite-Instruct...")
print("=" * 70)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="offload"
)

print("\nModel loaded successfully!")
print("=" * 70)
print("DeepSeek Programming Assistant")
print("Type 'exit' to quit.")
print("=" * 70)

SYSTEM_PROMPT = """
You are DeepSeek, an expert software engineering assistant.

Rules:
1. Answer ONLY programming, software engineering, algorithms, databases, operating systems, networking, cybersecurity, AI/ML, DevOps and computer science related questions.
2. If the question is unrelated to programming or computer science, politely reply:
   "I am a programming assistant and can only answer software and coding related questions."
3. Write clean, readable and efficient code.
4. Explain the solution before giving the code.
5. Mention the Time Complexity.
6. Mention the Space Complexity.
7. Use best coding practices.
8. If multiple approaches exist, recommend the best one.
9. Generate complete working code.
10. Use markdown code blocks.
"""

while True:

    user_prompt = input("\nYou : ").strip()

    if user_prompt.lower() in ["exit", "quit"]:
        print("\nDeepSeek : Goodbye!")
        break

    prompt = f"""
{SYSTEM_PROMPT}

User Question:
{user_prompt}

Assistant:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = response[len(prompt):].strip()

    print("\n" + "=" * 70)
    print("DeepSeek")
    print("=" * 70)
    print(answer)
    print("=" * 70)