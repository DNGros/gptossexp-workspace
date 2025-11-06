"""
Minimal reproducible example showing temperature=0 outputs differ between
Hugging Face Inference API and local transformers pipeline.

Both methods should produce identical outputs at temperature=0, but they don't.
This demonstrates a potential issue with how the pipeline applies chat templates
or generates responses.
"""

from huggingface_hub import InferenceClient
from transformers import pipeline
import json


def test_api(messages, max_tokens=200):
    """Test using Hugging Face Inference API."""
    print("="*70)
    print("API (InferenceClient)")
    print("="*70)
    
    client = InferenceClient()
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    
    response = completion.choices[0].message.content
    print(f"Response ({len(response)} chars):")
    print(response)
    print()
    return response


def test_pipeline(messages, max_tokens=200):
    """Test using local transformers pipeline."""
    print("="*70)
    print("LOCAL (transformers pipeline)")
    print("="*70)
    
    generator = pipeline(
        "text-generation",
        model="openai/gpt-oss-20b",
        torch_dtype="auto",
        device_map="auto",
    )
    
    result = generator(
        messages,
        max_new_tokens=max_tokens,
        temperature=1.0,  # pipeline requires temp > 0
        do_sample=False,  # but do_sample=False makes it deterministic
    )
    
    # Extract assistant's response
    generated = result[0]["generated_text"]
    if isinstance(generated, list) and len(generated) > len(messages):
        response = generated[-1].get("content", "")
    else:
        response = str(generated)
    
    print(f"Response ({len(response)} chars):")
    print(response)
    print()
    return response


def main():
    """Run the comparison test."""
    print("#"*70)
    print("TEMPERATURE=0 OUTPUTS DIFFER: API vs Pipeline")
    print("#"*70)
    print()
    
    # Simple deterministic prompt
    messages = [
        {
            "role": "user",
            "content": "Give some advice on planning a trip to Paris."
        }
    ]
    
    print("Messages:")
    print(json.dumps(messages, indent=2))
    print()
    
    # Test both methods
    api_response = test_api(messages, max_tokens=200)
    pipeline_response = test_pipeline(messages, max_tokens=200)
    
    # Compare
    print("="*70)
    print("COMPARISON")
    print("="*70)
    
    if api_response == pipeline_response:
        print("✓ Outputs are IDENTICAL")
    else:
        print("✗ Outputs DIFFER")
        print(f"\nAPI length: {len(api_response)} chars")
        print(f"Pipeline length: {len(pipeline_response)} chars")
        
        # Show first difference
        for i, (c1, c2) in enumerate(zip(api_response, pipeline_response)):
            if c1 != c2:
                print(f"\nFirst difference at position {i}:")
                print(f"  API: ...{api_response[max(0,i-20):i+20]}...")
                print(f"  Pipeline: ...{pipeline_response[max(0,i-20):i+20]}...")
                break


if __name__ == "__main__":
    main()
