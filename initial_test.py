from Prompt_Builder.prompt_builder import build_prompt
from Generator.generation import generate_text

# Define your params here
params = {
    "gender": "Male",
    "race": "Human",
    "age": "40",
    "background": "A bartender that is a war veteran, haunted by his past.",
}

# Path to your Jinja2 template
template_path = "npc.j2"

# Build the prompt using your template and params
prompt = build_prompt(template_path, params)

# Generate the response using your LLM function
response = generate_text(prompt)

# Output the response
print(response)