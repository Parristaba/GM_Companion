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

def generate_text(prompt: str) -> str:
    output = generator(prompt, return_full_text=False)[0]["generated_text"]
    return output
