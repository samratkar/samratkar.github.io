import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError

# Configure UTF-8 output on Windows consoles to prevent charmap errors with emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load environment variables from parent folder (claude-academy) or local directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(f"ANTHROPIC_API_KEY not found in {env_path}. Please check your .env file.")

client = anthropic.Anthropic(api_key=api_key)

# 2. Define tools using Pydantic Models for schema generation & input validation
class GetWeather(BaseModel):
    """Get the current weather for a city."""
    city: str = Field(description="The city name to get weather for, e.g. 'Austin'")

    def execute(self) -> str:
        # In a real app, this would call a real weather API
        return f"Weather in {self.city}: 95°F, sunny with clear skies."

class GetForecast(BaseModel):
    """Get the multi-day weather forecast for a city."""
    city: str = Field(description="The city name to get forecast for, e.g. 'Austin'")
    days: int = Field(default=3, description="Number of days to forecast (1 to 7)", ge=1, le=7)

    def execute(self) -> str:
        # In a real app, this would call a forecast API
        return (
            f"Forecast for {self.city} ({self.days} days):\n"
            f"- Tomorrow: 90°F, partly cloudy\n"
            f"- Day 2: 82°F, scattered showers and thunderstorms\n"
            f"- Day 3: 78°F, light rain clearing by evening"
        )

# 3. Tool Registry mapping tool name -> Pydantic model
TOOL_REGISTRY = {
    "get_weather": GetWeather,
    "get_forecast": GetForecast,
}

# 4. Auto-generate Claude tool schemas directly from Pydantic
tools = [
    {
        "name": name,
        "description": model_cls.__doc__ or "",
        "input_schema": model_cls.model_json_schema(),
    }
    for name, model_cls in TOOL_REGISTRY.items()
]

# 5. Type-safe tool runner with Pydantic validation
def run_tool(name: str, tool_input: dict) -> str:
    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'"
    
    model_cls = TOOL_REGISTRY[name]
    try:
        # Pydantic validates types and required fields at runtime
        instance = model_cls(**tool_input)
        return instance.execute()
    except ValidationError as e:
        # Return validation errors back to Claude so it can self-correct!
        return f"Validation error for tool '{name}': {e}"

def main():
    messages = [
        {"role": "user", "content": "What should I wear in Austin today, and will I need an umbrella over the next few days?"}
    ]

    print("--- Starting Agent Loop ---\n")

    # 6. Agent Loop: sends messages to Claude until end_turn
    while True:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Claude is finished thinking & responding
            for block in response.content:
                if block.type == "text":
                    print("--- Claude's Final Response ---")
                    print(block.text)
            break

        if response.stop_reason == "tool_use":
            # Claude requested one or more tool calls
            tool_results = []
            for block in response.content:
                if block.type == "text":
                    print(f"[Claude Thinking]: {block.text}")
                elif block.type == "tool_use":
                    print(f"[Tool Call Request]: {block.name} with inputs {block.input}")
                    result = run_tool(block.name, block.input)
                    print(f"[Tool Result]: {result}\n")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # Push assistant response and tool results back into history for next iteration
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    main()