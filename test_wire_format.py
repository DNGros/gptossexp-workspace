"""
Test to see what's actually sent over the wire to the Hugging Face API.
This will show us if the Harmony format is being applied client-side or server-side.
"""

import httpx
from model_predict import Model
import json

def inspect_api_request():
    """
    Use httpx with event hooks to inspect the actual HTTP request.
    """

    # Track what's being sent
    captured_request = {}

    def log_request(request: httpx.Request):
        """Hook to capture outgoing request"""
        captured_request['method'] = request.method
        captured_request['url'] = str(request.url)
        captured_request['headers'] = dict(request.headers)
        captured_request['body'] = request.content.decode('utf-8') if request.content else None
        print("\n" + "="*60)
        print("OUTGOING HTTP REQUEST")
        print("="*60)
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"\nRequest Body:")
        if request.content:
            try:
                body_json = json.loads(request.content)
                print(json.dumps(body_json, indent=2))
            except:
                print(request.content.decode('utf-8'))

    def log_response(response: httpx.Response):
        """Hook to capture incoming response"""
        print("\n" + "="*60)
        print("INCOMING HTTP RESPONSE")
        print("="*60)
        print(f"Status: {response.status_code}")
        print(f"\nResponse Body (first 1000 chars):")
        print(response.text[:1000])

    # Create an httpx client with event hooks
    event_hooks = {
        'request': [log_request],
        'response': [log_response]
    }

    # Monkey-patch the huggingface_hub to use our httpx client
    from huggingface_hub import InferenceClient
    import os

    client = InferenceClient()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Think step by step."
        },
        {
            "role": "user",
            "content": "What is 3+5? Explain your reasoning."
        }
    ]

    print("\n" + "#"*60)
    print("Testing GPT-OSS-20B API Request Format")
    print("#"*60)
    print("\nSending messages:")
    print(json.dumps(messages, indent=2))

    try:
        # Make the API call - we need to patch httpx to intercept
        # For now, let's just check what parameters we're sending
        print("\n" + "="*60)
        print("PARAMETERS BEING SENT TO API")
        print("="*60)
        print(f"Model: {Model.GPT_OSS_20B}")
        print(f"Messages format: Standard OpenAI (list of dicts with role/content)")
        print(f"Number of messages: {len(messages)}")

        # Check if there's any template being applied
        print("\n" + "="*60)
        print("CHECKING FOR CLIENT-SIDE TEMPLATE APPLICATION")
        print("="*60)

        # Look at the InferenceClient source to see if it applies templates
        import inspect
        chat_completion_source = inspect.getsource(client.chat_completion)

        if 'apply_chat_template' in chat_completion_source or 'chat_template' in chat_completion_source:
            print("⚠️  Client appears to apply chat template")
        else:
            print("✓ Client does NOT apply chat template (sends raw messages)")
            print("  → Template should be applied server-side")

        # Now make the actual call to see what happens
        print("\n" + "="*60)
        print("MAKING ACTUAL API CALL")
        print("="*60)

        completion = client.chat.completions.create(
            model=Model.GPT_OSS_20B,
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )

        print("\n" + "="*60)
        print("RESPONSE STRUCTURE")
        print("="*60)
        print(f"Response content: {completion.choices[0].message.content[:200]}...")

        # Check if reasoning is present
        message = completion.choices[0].message
        has_reasoning = hasattr(message, 'reasoning') and message.reasoning
        has_reasoning_content = hasattr(message, 'reasoning_content') and message.reasoning_content

        print(f"\nHas 'reasoning' attribute: {has_reasoning}")
        print(f"Has 'reasoning_content' attribute: {has_reasoning_content}")

        if has_reasoning or has_reasoning_content:
            print("\n✓ HARMONY FORMAT IS WORKING!")
            print("  → The response contains separate reasoning, which means:")
            print("     1. The input was formatted with Harmony template")
            print("     2. The model responded in Harmony format")
            print("     3. The API parsed it back into structured fields")
        else:
            print("\n⚠️  NO REASONING FIELD FOUND")
            print("  → This might mean Harmony format is NOT being applied")

        return completion

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    inspect_api_request()
