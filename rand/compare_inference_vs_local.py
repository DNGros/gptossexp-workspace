"""
Compare responses from Hugging Face Inference API vs local Transformers.

This script tests whether both methods apply the Harmony format correctly by:
1. Calling the same prompt via InferenceClient (API)
2. Calling the same prompt via local Transformers with apply_chat_template
3. Comparing the outputs to verify they're similar

Uses temperature=0 for determinism and a math problem that requires reasoning
but has a deterministic answer.
"""

from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from model_predict import Model
from difflib import SequenceMatcher


def test_inference_client(messages, max_tokens=800):
    """Test using Hugging Face Inference API (what model_predict.py uses)."""
    print("\n" + "="*70)
    print("METHOD 1: Hugging Face Inference Client (API)")
    print("="*70)

    client = InferenceClient()

    print(f"Model: {Model.GPT_OSS_20B}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print("\nCalling API...")

    completion = client.chat.completions.create(
        model=Model.GPT_OSS_20B,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    message = completion.choices[0].message

    # Try to get reasoning
    reasoning = None
    for attr in ['reasoning', 'reasoning_content']:
        if hasattr(message, attr):
            val = getattr(message, attr)
            if val:
                reasoning = val
                print(f"\nFound reasoning in attribute: '{attr}'")
                break

    result = {
        'content': message.content,
        'reasoning': reasoning,
        'method': 'InferenceClient'
    }

    print(f"\nResponse content ({len(message.content)} chars):")
    print(message.content[:300] + ("..." if len(message.content) > 300 else ""))

    if reasoning:
        print(f"\nReasoning ({len(reasoning)} chars):")
        print(reasoning[:300] + ("..." if len(reasoning) > 300 else ""))
    else:
        print("\n⚠️  No reasoning found")

    return result


def test_local_transformers(messages, max_tokens=800):
    """Test using local Transformers with apply_chat_template (Harmony format)."""
    print("\n" + "="*70)
    print("METHOD 2: Local Transformers with apply_chat_template")
    print("="*70)

    model_id = "openai/gpt-oss-20b"

    print(f"Loading model: {model_id}")
    print("(This may take a while on first run...)")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Model loaded on device: {next(model.parameters()).device}")

    # Apply chat template (this should apply Harmony format)
    print(f"\nMessages: {json.dumps(messages, indent=2)}")
    print("\nApplying chat template...")

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # Show what the template produced
    input_text = tokenizer.decode(inputs['input_ids'][0])
    print(f"\nTemplated input ({len(input_text)} chars):")
    print(input_text[:500] + ("..." if len(input_text) > 500 else ""))

    # Check for Harmony format markers
    harmony_markers = ['<|channel|>', '<|analysis|>', '<|final|>', '<|message|>']
    found_markers = [m for m in harmony_markers if m in input_text]
    if found_markers:
        print(f"✓ Found Harmony format markers: {found_markers}")
    else:
        print("⚠️  No Harmony format markers found in input")

    # Generate response
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # temperature=0 equivalent
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the new tokens (the response)
    response_only = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    print(f"\nRaw response ({len(response_only)} chars):")
    print(response_only[:500] + ("..." if len(response_only) > 500 else ""))

    # Try to parse Harmony format channels
    # The response might have <|channel|>analysis and <|channel|>final sections
    reasoning = None
    content = None

    # Simple parsing - look for analysis channel and final channel
    if '<|channel|>analysis' in response_only:
        print("\n✓ Found <|channel|>analysis in response")
        # Extract content between <|channel|>analysis and next <|channel|> or <|end|>
        parts = response_only.split('<|channel|>')
        for i, part in enumerate(parts):
            if part.startswith('analysis'):
                # Get text after 'analysis<|message|>'
                if '<|message|>' in part:
                    analysis_content = part.split('<|message|>', 1)[1]
                    # Stop at next special token
                    analysis_content = analysis_content.split('<|')[0]
                    reasoning = analysis_content.strip()
                    print(f"Extracted reasoning: {reasoning[:200]}...")

    if '<|channel|>final' in response_only:
        print("✓ Found <|channel|>final in response")
        parts = response_only.split('<|channel|>')
        for i, part in enumerate(parts):
            if part.startswith('final'):
                if '<|message|>' in part:
                    final_content = part.split('<|message|>', 1)[1]
                    final_content = final_content.split('<|')[0]
                    content = final_content.strip()
                    print(f"Extracted final content: {content[:200]}...")

    # Fallback: if we can't parse channels, use the whole response with special tokens removed
    if content is None:
        content = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print("⚠️  Could not parse Harmony channels, using full decoded response")

    result = {
        'content': content,
        'reasoning': reasoning,
        'method': 'Local Transformers',
        'raw_output': response_only
    }

    print(f"\nFinal content ({len(content)} chars):")
    print(content[:300] + ("..." if len(content) > 300 else ""))

    if reasoning:
        print(f"\nReasoning ({len(reasoning)} chars):")
        print(reasoning[:300] + ("..." if len(reasoning) > 300 else ""))

    return result


def compare_results(result1, result2):
    """Compare the two results."""
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    # Compare content
    content1 = result1['content'] or ""
    content2 = result2['content'] or ""

    content_similarity = SequenceMatcher(None, content1, content2).ratio()

    print(f"\nContent similarity: {content_similarity:.1%}")
    print(f"  Method 1 length: {len(content1)} chars")
    print(f"  Method 2 length: {len(content2)} chars")

    if content_similarity > 0.8:
        print("  ✓ Highly similar - formats likely consistent")
    elif content_similarity > 0.5:
        print("  ⚠️  Moderately similar - some differences")
    else:
        print("  ❌ Very different - possible format mismatch")

    # Compare reasoning
    reasoning1 = result1['reasoning'] or ""
    reasoning2 = result2['reasoning'] or ""

    if reasoning1 and reasoning2:
        reasoning_similarity = SequenceMatcher(None, reasoning1, reasoning2).ratio()
        print(f"\nReasoning similarity: {reasoning_similarity:.1%}")
        print(f"  Method 1 length: {len(reasoning1)} chars")
        print(f"  Method 2 length: {len(reasoning2)} chars")

        if reasoning_similarity > 0.8:
            print("  ✓ Highly similar")
        elif reasoning_similarity > 0.5:
            print("  ⚠️  Moderately similar")
        else:
            print("  ❌ Very different")
    elif reasoning1 and not reasoning2:
        print("\n⚠️  Method 1 has reasoning, Method 2 does not")
    elif reasoning2 and not reasoning1:
        print("\n⚠️  Method 2 has reasoning, Method 1 does not")
    else:
        print("\n⚠️  Neither method extracted reasoning")

    # Show side-by-side comparison
    print("\n" + "-"*70)
    print("SIDE-BY-SIDE CONTENT (first 200 chars)")
    print("-"*70)
    print(f"Method 1: {content1[:200]}")
    print(f"Method 2: {content2[:200]}")

    if reasoning1 or reasoning2:
        print("\n" + "-"*70)
        print("SIDE-BY-SIDE REASONING (first 200 chars)")
        print("-"*70)
        print(f"Method 1: {reasoning1[:200] if reasoning1 else '(none)'}")
        print(f"Method 2: {reasoning2[:200] if reasoning2 else '(none)'}")


def main():
    """Run the comparison test."""
    print("\n" + "#"*70)
    print("COMPARING INFERENCE CLIENT VS LOCAL TRANSFORMERS")
    print("#"*70)

    # Create a prompt that:
    # - Has a deterministic answer (math problem)
    # - Requires reasoning
    # - Requires creative output (story) to produce longer, more comparable responses
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who loves math and creative storytelling. Always show your reasoning step-by-step."
        },
        {
            "role": "user",
            "content": """First, calculate 17 * 23, then add 156. Show your work step by step.

Then, using the final answer as the number of goats, write a short 3-paragraph story about a farmer who discovers exactly that many goats have mysteriously appeared on their farm one morning. Make it whimsical and include the exact number prominently in the story."""
        }
    ]

    try:
        # Test method 1: Inference Client
        result1 = test_inference_client(messages, max_tokens=800)

        # Test method 2: Local Transformers
        result2 = test_local_transformers(messages, max_tokens=800)

        # Compare results
        compare_results(result1, result2)

        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print("\nConclusion:")
        print("If the outputs are highly similar, both methods apply Harmony format correctly.")
        print("If they differ significantly, there may be a formatting issue.")

    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
