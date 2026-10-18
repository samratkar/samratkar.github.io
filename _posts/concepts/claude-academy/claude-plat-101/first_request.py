import os
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
    print("Sending first request to Claude API...\n")

    # Make the first API request using the Messages API
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": "Hello Claude! In two sentences, introduce yourself to someone learning the Anthropic API."
            }
        ]
    )

    print("--- Claude's Response ---")
    for block in message.content:
        if block.type == "text":
            print(block.text)

    print("\n--- Metadata ---")
    print(f"Model used       : {message.model}")
    print(f"Stop reason      : {message.stop_reason}")
    print(f"Input tokens     : {message.usage.input_tokens}")
    print(f"Output tokens    : {message.usage.output_tokens}")

if __name__ == "__main__":
    main()
