import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(f"ANTHROPIC_API_KEY not found in {env_path}. Please check your .env file.")

client = anthropic.Anthropic(api_key=api_key)


# 2. Define tools using official @anthropic.beta_tool
# (This is the Python equivalent of TypeScript's betaZodTool - schemas & validation are 100% automatic via Pydantic)
@anthropic.beta_tool
def get_weather(city: str) -> str:
    """Get today's current weather for a city."""
    return f"Weather in {city}: 95°F, sunny with clear skies."


@anthropic.beta_tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get the weather forecast for the next few days for a city."""
    return (
        f"Forecast for {city} ({days} days):\n"
        f"- Tomorrow: 90°F, partly cloudy\n"
        f"- Day 2: 82°F, scattered showers and thunderstorms\n"
        f"- Day 3: 78°F, light rain clearing by evening"
    )


def main():
    prompt = "I'm packing for a three-day trip to Denver. What's the weather today and over the next few days?"
    print("--- Running with Official Anthropic tool_runner ---\n")

    # 3. Official built-in Tool Runner
    # (Exact equivalent of TypeScript's client.beta.messages.toolRunner)
    runner = client.beta.messages.tool_runner(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tools=[get_weather, get_forecast],
    )

    # 4. until_done() automatically handles the ping-pong agent loop
    # (Exact equivalent of TypeScript's `await runner`)
    final_message = runner.until_done()

    print("--- Claude's Final Response ---")
    for block in final_message.content:
        if block.type == "text":
            print(block.text)


if __name__ == "__main__":
    main()
