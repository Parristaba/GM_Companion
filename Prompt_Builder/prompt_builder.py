from jinja2 import Environment, FileSystemLoader
import os

# Define where your templates are stored
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Initialize Jinja2 environment
env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=False)

def build_prompt(template_name: str, params: dict) -> str:
    """
    Renders a prompt using the specified Jinja2 template and parameters.

    Args:
        template_name (str): Name of the Jinja2 template file.
        params (dict): Parameters to inject into the template.

    Returns:
        str: The rendered prompt as a string.
    """
    template = env.get_template(template_name)
    return template.render(**params)
