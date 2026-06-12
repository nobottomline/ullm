"""
Guaranteed tool calling — the building block of an agent.

You give the model OpenAI-style `tools`; uLLM constrains decoding so the reply
is *always* a valid call of one of them — the right function name and arguments
that match its JSON Schema. No malformed tool calls, ever.

Run:
    ullm serve /path/to/model.gguf --gpu      # in another terminal
    pip install openai
    python examples/tool_call.py
"""

import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="ullm")


def get_weather(location, unit="celsius"):
    # A real tool would call an API; here we fake it.
    return f"22 degrees {unit}, sunny, in {location}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    }
]

# 1) The model picks a tool and fills its arguments — guaranteed valid.
resp = client.chat.completions.create(
    model="ullm",
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris, in celsius?"}],
)
call = resp.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)  # always valid JSON, matching the schema
print(f"→ {call.function.name}({args})")

# 2) Run the tool and let the model answer with the result.
observation = get_weather(**args)
final = client.chat.completions.create(
    model="ullm",
    messages=[
        {"role": "user", "content": "What's the weather in Paris, in celsius?"},
        {"role": "assistant", "content": f"(I checked: {observation})"},
        {"role": "user", "content": "Now answer me in one sentence."},
    ],
)
print(final.choices[0].message.content.strip())
