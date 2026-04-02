from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

model_name = "google/gemma-2b"
save_path = "models/gemma-2b"

# Create directory
os.makedirs(save_path, exist_ok=True)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Save locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"✅ Model saved at: {save_path}")