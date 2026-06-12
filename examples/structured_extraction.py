"""
Guaranteed structured extraction — a drop-in local OpenAI.

uLLM's server speaks the OpenAI API, but with `response_format` it *guarantees*
the reply matches your JSON Schema: no retries, no "please respond in JSON", no
JSON-repair. The same code points at OpenAI by changing base_url.

Run:
    ullm serve /path/to/model.gguf --gpu      # in another terminal
    pip install openai
    python examples/structured_extraction.py
"""

import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="ullm")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

resp = client.chat.completions.create(
    model="ullm",
    messages=[
        {"role": "user", "content": "Extract the person: Sarah is 27 and lives in Berlin."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "person", "schema": schema},
    },
)

# Guaranteed to parse and match the schema.
person = json.loads(resp.choices[0].message.content)
print(person)
assert set(person) >= {"name", "age", "city"} and isinstance(person["age"], int)
print("✓ conforms to the schema")
