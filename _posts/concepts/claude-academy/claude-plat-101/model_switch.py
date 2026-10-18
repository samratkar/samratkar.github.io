import os
import time
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from parent folder (claude-academy) or local directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Verify key is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(f"ANTHROPIC_API_KEY not found in {env_path}. Please check your .env file.")

# Initialize Anthropic client
client = Anthropic(api_key=api_key)

def main():
    models = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5"]
    prompt = "Hello Claude! In two sentences, introduce yourself to someone learning the Anthropic API."

    for model in models:
        start_time = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Safely extract text (some models like Opus return ThinkingBlock first)
        text_blocks = [b.text for b in response.content if b.type == "text"]
        reply = "\n".join(text_blocks).strip()
        
        print(f"[{model}]  {elapsed_ms}ms  in={response.usage.input_tokens} out={response.usage.output_tokens}")
        print(reply)
        print()

if __name__ == "__main__":
    main()
