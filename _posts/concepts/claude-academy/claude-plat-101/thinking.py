import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Configure UTF-8 output on Windows consoles
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

# 2. Define tools for weather and forecast
weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Monterey' or 'Napa'"}
        },
        "required": ["city"],
    },
}

forecast_tool = {
    "name": "get_forecast",
    "description": "Get the multi-day weather forecast for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Number of days (1 to 5)"}
        },
        "required": ["city"],
    },
}

tools = [weather_tool, forecast_tool]

# Mock tool runner returning realistic data for the road trip stops
def run_tool(name: str, tool_input: dict) -> str:
    city = tool_input.get("city", "Unknown")
    city_lower = city.lower()
    
    if name == "get_weather":
        if "monterey" in city_lower or "carmel" in city_lower:
            return f"Weather in {city}: 64°F, mild, light coastal fog clearing by noon."
        elif "napa" in city_lower or "sonoma" in city_lower:
            return f"Weather in {city}: 78°F, warm and sunny with clear blue skies."
        elif "tahoe" in city_lower:
            return f"Weather in {city}: 52°F, crisp and sunny, dropping to 34°F at night."
        else:
            return f"Weather in {city}: 70°F, comfortable and partly cloudy."

    if name == "get_forecast":
        days = tool_input.get("days", 3)
        return (
            f"Forecast for {city} ({days} days):\n"
            f"- Day 1: 72°F, clear skies\n"
            f"- Day 2: 68°F, breezy with afternoon sun\n"
            f"- Day 3: 65°F, mild and pleasant"
        )

    raise ValueError(f"Unknown tool: {name}")

def main():
    prompt = (
        "Plan a weekend road trip out of San Francisco with two scenic stops, "
        "weighing weather conditions, drive time, and outdoor activities."
    )

    print("=================================================================")
    print("  Claude Extended Thinking (Adaptive Reasoning) Feature Demo")
    print("=================================================================\n")
    print(f"User Request: {prompt}\n")

    messages = [{"role": "user", "content": prompt}]

    # The Thinking Agent Loop
    while True:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=16000,
            thinking={"type": "adaptive", "display": "summarized"},  # Adaptive reasoning with summarized thoughts
            output_config={"effort": "high"},                       # Effort: low | medium | high | xhigh | max
            tools=tools,
            messages=messages,
        )

        # Inspect and print reasoning blocks if returned
        for block in response.content:
            if block.type == "thinking":
                print("🧠 [Claude's Internal Thinking Process]:")
                print("-" * 65)
                print(block.thinking.strip() if hasattr(block, "thinking") and block.thinking else "(Thinking performed internally)")
                print("-" * 65 + "\n")

        # Termination: Claude finished its complete response
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print("✨ [Claude's Final Plan]:")
                    print("=" * 65)
                    print(block.text.strip())
                    print("=" * 65 + "\n")
            break

        # Tool Use: Claude called one or more tools during its deliberation
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"🛠️ [Tool Call]: {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    print(f"   ↳ [Tool Result]: {result}\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    # Display usage & reasoning token metrics
    print("📊 [Usage & Thinking Metrics]:")
    print(f"Model used       : {response.model}")
    print(f"Input tokens     : {response.usage.input_tokens}")
    print(f"Output tokens    : {response.usage.output_tokens}")
    if hasattr(response.usage, "output_tokens_details") and response.usage.output_tokens_details:
        thinking_toks = getattr(response.usage.output_tokens_details, "thinking_tokens", 0)
        print(f"Thinking tokens  : {thinking_toks}")


if __name__ == "__main__":
    main()
