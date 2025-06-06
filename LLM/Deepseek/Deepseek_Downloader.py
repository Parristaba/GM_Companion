from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
save_directory = os.path.join("LLM", "Deepseek", model_name.replace("/", "_"))

os.makedirs(save_directory, exist_ok=True)

# Load the model and tokenizer from Huggingface (it will download automatically)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Save the model and tokenizer to a local directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)


print(f"Model '{model_name}' has been downloaded to '{save_directory}'")