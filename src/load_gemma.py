from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path = "models/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    device_map="auto"
)

prompt = "What is artificial intelligence?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== MODEL OUTPUT ===\n")
print(response)