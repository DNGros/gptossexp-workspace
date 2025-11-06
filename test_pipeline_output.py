"""
Test script to see what the transformers pipeline returns.

This will help us understand:
1. What format the output is in
2. Whether reasoning is included
3. Whether Harmony tokens are visible
"""

from transformers import pipeline
import json
from .model_predict import get_model_response, InferenceBackend


def test_pipeline_output():
    """Test the pipeline with a simple prompt and inspect the output."""
    print("="*70)
    print("TESTING TRANSFORMERS PIPELINE OUTPUT")
    print("="*70)
    
    # Test with a simple prompt that should produce reasoning
    messages = [
        {"role": "system", "content": "You are a helpful math tutor. Show your work."},
        {"role": "user", "content": "What is 17 * 23? Show your reasoning step by step."},
    ]
    
    print("Input messages:")
    print(json.dumps(messages, indent=2))

    print(get_model_response(
        model="openai/gpt-oss-20b",
        prompt="What is 17 * 23? Show your reasoning step by step.",
        system_prompt="You are a helpful math tutor. Show your work.",
        max_tokens=5000,
        temperature=0,
        use_cache=False,
        backend=InferenceBackend.LOCAL,
    ))
    exit()


    print("\nLoading pipeline (this may take a while)...")
    
    # Create pipeline
    generator = pipeline(
        "text-generation",
        model="openai/gpt-oss-20b",
        torch_dtype="auto",
        device_map="auto",
    )
    
    print("✓ Pipeline loaded\n")

    print("\nGenerating response...")
    
    # Generate
    result = generator(
        messages,
        max_new_tokens=5000,
        temperature=0.7,
    )
    
    print("\n" + "="*70)
    print("RAW OUTPUT INSPECTION")
    print("="*70)
    
    # Inspect the raw result
    print(f"\nType of result: {type(result)}")
    print(f"Length of result: {len(result)}")
    print(f"\nType of result[0]: {type(result[0])}")
    print(f"Keys in result[0]: {result[0].keys()}")
    
    # Look at generated_text
    generated_text = result[0]["generated_text"]
    print(f"\nType of generated_text: {type(generated_text)}")
    
    if isinstance(generated_text, list):
        print(f"Length of generated_text: {len(generated_text)}")
        print(f"\nNumber of input messages: {len(messages)}")
        print(f"Number of output messages: {len(generated_text)}")
        
        print("\n" + "-"*70)
        print("ALL MESSAGES IN OUTPUT:")
        print("-"*70)
        for i, msg in enumerate(generated_text):
            print(f"\nMessage {i}:")
            print(f"  Role: {msg.get('role', 'N/A')}")
            content = msg.get('content', '')
            if len(content) > 1000:
                print(f"  Content: {content[:1000]}...")
                print(f"  (truncated, full length: {len(content)} chars)")
            else:
                print(f"  Content: {content}")
        
        # Extract just the assistant's response
        if len(generated_text) > len(messages):
            assistant_msg = generated_text[-1]
            print("\n" + "="*70)
            print("EXTRACTED ASSISTANT RESPONSE:")
            print("="*70)
            print(f"Role: {assistant_msg.get('role', 'N/A')}")
            print(f"Content:\n{assistant_msg.get('content', '')}")
            
            # Check for Harmony tokens
            content = assistant_msg.get('content', '')
            harmony_markers = ['<|channel|>', '<|analysis|>', '<|final|>', '<|message|>']
            found_markers = [m for m in harmony_markers if m in content]
            
            if found_markers:
                print(f"\n⚠️  Found Harmony format markers: {found_markers}")
                print("The pipeline is NOT hiding Harmony tokens")
            else:
                print("\n✓ No Harmony format markers found")
                print("The pipeline is hiding Harmony tokens (clean output)")
    else:
        print(f"\nGenerated text (string):\n{generated_text}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        test_pipeline_output()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

