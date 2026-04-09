from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_name = "google/gemma-2b"
save_path = "models/gemma-2b"

os.makedirs(save_path, exist_ok=True)

# Load FULL model (no quantization)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# Save safely
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"✅ Model saved at: {save_path}")