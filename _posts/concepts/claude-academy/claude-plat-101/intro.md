---
layout: post
title: "Claude Platform 101: Making Your First API Request"
type: concepts
category: claude-academy
date: 2026-10-18
---

## Introduction

In this module of **Claude Platform 101**, we configure our API environment and send our very first request to the Anthropic Messages API using Python.

---

## 1. Prerequisites & Environment Setup

### 1.1 Secure API Key Storage

Store your Anthropic API key in a `.env` file in the parent `claude-academy` directory so it can be shared across all academy modules:

```env
ANTHROPIC_API_KEY=your-api-key-here
```

> **Note**: Ensure `.env` is listed in your `.gitignore` to prevent leaking credentials.

### 1.2 Install Dependencies

Using the shared virtual environment in `claude-academy/.venv`:

```bash
# In claude-academy/
python -m venv .venv
.venv\Scripts\pip install anthropic python-dotenv
```

---

## 2. First Request Script (`first_request.py`)

Here is the complete Python code to initialize the Anthropic client, construct a message, and print Claude's response along with usage metadata:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from parent folder (claude-academy) or local directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Verify key exists
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
```

---

## 3. Running the Script

Run the script from your terminal:

```bash
python first_request.py
```

### Example Output

```text
Sending first request to Claude API...

--- Claude's Response ---
I'm Claude, an AI assistant made by Anthropic. I'm here to help you build applications through the Anthropic API by answering questions, explaining concepts, and providing guidance on how to integrate me into your projects.

--- Metadata ---
Model used       : claude-haiku-4-5-20251001
Stop reason      : end_turn
Input tokens     : 25
Output tokens    : 50
```
