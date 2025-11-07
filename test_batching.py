"""
Quick test script to verify batching implementation works correctly.
"""

from workspace.model_predict import Model, InferenceBackend
from workspace.run_toxic_chat import run_toxic_chat_evaluation

def test_batching():
    """Test batching with a small sample."""
    print("="*80)
    print("Testing batching implementation with LOCAL backend")
    print("="*80)

    # Run with LOCAL backend and batching
    print("\n1. Running with LOCAL backend (batch_size=4)")
    print("-" * 80)

    result_df = run_toxic_chat_evaluation(
        sample_size=10,  # Small sample for quick test
        version='toxicchat0124',
        split='test',
        policy='toxic_chat_claude_1',
        model='GPT_OSS_20B',
        use_cache=True,
        backend=InferenceBackend.LOCAL,
        batch_size=4,
    )

    print(f"\n✓ Successfully processed {len(result_df)} examples")
    print(f"✓ Columns in result: {list(result_df.columns)}")

    # Check that backend column was added
    if 'backend' in result_df.columns:
        print(f"✓ Backend column present: {result_df['backend'].iloc[0]}")
    else:
        print("⚠ Warning: Backend column not found (expected for LOCAL backend)")

    # Show sample results
    print("\nSample results:")
    print(result_df[['user_input', 'predicted_toxic', 'parsed_successfully']].head())

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)

    return result_df


if __name__ == "__main__":
    test_batching()
