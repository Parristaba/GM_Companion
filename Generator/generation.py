from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model with 4-bit quantization
model_name_or_path = r"C:\Users\kagan_ntaijui\Desktop\GM_Companion\LLM\Deepseek\Deepseek_1.5b_Qwen"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    load_in_4bit=True,              # this does 4-bit quantization on load
    trust_remote_code=True
)

# Create generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1600,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

import re
import json

def extract_json(output: str) -> dict:
    """
    Extracts the first JSON block from the LLM output.
    Returns the parsed dict or raises ValueError if extraction fails.
    """
    # Try to find ```json ... ```
    match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
    if not match:
        # Fallback: find any {...} in text
        match = re.search(r'(\{.*?\})', output, re.DOTALL)
    
    if not match:
        raise ValueError("No JSON found in model output.")
    
    json_str = match.group(1)

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {e}")


def generate_text(prompt: str) -> str:
    output = generator(prompt, return_full_text=False)[0]["generated_text"]
    try:
        # Attempt to extract JSON from the output
        json_data = extract_json(output)
        return json_data
    except ValueError as e:
        # If extraction fails, return the raw output
        print(f"Warning: {e}")
    return output
