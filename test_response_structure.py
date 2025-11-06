"""
Test script to inspect the actual response structure from GPT-OSS models.
This will help verify if reasoning is being returned and what attribute name to use.
"""

from huggingface_hub import InferenceClient
from model_predict import Model
import json

def inspect_response_structure():
    """Inspect what's actually in the API response."""
    client = InferenceClient()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Think step by step."
        },
        {
            "role": "user",
            "content": "What is 15 * 24? Show your reasoning."
        }
    ]

    print("Making API call to GPT-OSS-20B...")
    completion = client.chat.completions.create(
        model=Model.GPT_OSS_20B,
        messages=messages,
        temperature=0.0,
        max_tokens=1000,
    )

    print("\n" + "="*60)
    print("RESPONSE STRUCTURE INSPECTION")
    print("="*60)

    # Check what's in the completion object
    print("\nCompletion object attributes:")
    print(dir(completion))

    # Check the first choice
    print("\n" + "-"*60)
    print("choices[0] attributes:")
    print(dir(completion.choices[0]))

    # Check the message
    print("\n" + "-"*60)
    print("choices[0].message attributes:")
    message = completion.choices[0].message
    print(dir(message))

    # Try to get the actual values
    print("\n" + "-"*60)
    print("Message content:")
    print(message.content)

    # Check for reasoning-related attributes
    print("\n" + "-"*60)
    print("Checking for reasoning attributes:")

    possible_reasoning_attrs = [
        'reasoning',
        'reasoning_content',
        'thought',
        'chain_of_thought',
        'analysis'
    ]

    for attr in possible_reasoning_attrs:
        if hasattr(message, attr):
            value = getattr(message, attr)
            print(f"  ✓ Found '{attr}': {value[:200] if value else 'None'}...")
        else:
            print(f"  ✗ No '{attr}' attribute")

    # Try to convert to dict if possible
    print("\n" + "-"*60)
    print("Attempting to serialize to dict:")
    try:
        if hasattr(message, 'model_dump'):
            print(json.dumps(message.model_dump(), indent=2))
        elif hasattr(message, 'dict'):
            print(json.dumps(message.dict(), indent=2))
        elif hasattr(message, '__dict__'):
            print(json.dumps(message.__dict__, indent=2, default=str))
        else:
            print("Cannot serialize to dict")
    except Exception as e:
        print(f"Error serializing: {e}")

    # Check the full completion object
    print("\n" + "-"*60)
    print("Full completion object:")
    try:
        if hasattr(completion, 'model_dump'):
            print(json.dumps(completion.model_dump(), indent=2))
        elif hasattr(completion, 'dict'):
            print(json.dumps(completion.dict(), indent=2))
    except Exception as e:
        print(f"Error serializing completion: {e}")


if __name__ == "__main__":
    inspect_response_structure()
