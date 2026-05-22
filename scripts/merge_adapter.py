import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "models/llama-1b-base"
ADAPTER    = "models/llama-1b-mental"
OUTPUT     = "models/llama-1b-mental-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUTPUT)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.save_pretrained(OUTPUT)

print(f"Done! Merged model saved to {OUTPUT}")
