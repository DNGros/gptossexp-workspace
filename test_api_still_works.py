"""
Quick test to verify that existing API functionality still works.
This uses the API backend (default) to ensure backward compatibility.
"""

def test_api_backend():
    """Test that API backend still works (existing functionality)."""
    print("Testing API backend (backward compatibility)...")
    
    from workspace.model_predict import get_model_response, Model
    
    # Simple test that should work quickly
    response = get_model_response(
        model=Model.GPT_OSS_20B,
        prompt="Say 'test successful' and nothing else.",
        use_cache=False,  # Don't use cache for this test
    )
    
    print(f"\nResponse: {response.response}")
    print(f"Reasoning: {response.reasoning[:100] if response.reasoning else '(none)'}...")
    print(f"Model: {response.model}")
    
    assert response.response, "Response should not be empty"
    assert response.model == Model.GPT_OSS_20B, "Model should match"
    
    print("\n✓ API backend works correctly")
    return True


def test_classify_api():
    """Test that classify still works with default (API) backend."""
    print("\nTesting classify with API backend...")
    
    from workspace.classify import classify
    from workspace.model_predict import Model
    from workspace.policies import toxic_simple
    
    # Test with a clearly non-toxic message
    result = classify(
        text="Hello, how are you today?",
        policy_module=toxic_simple,
        model=Model.GPT_OSS_20B,
        # backend defaults to API
    )
    
    print(f"\nBinary label: {result.binary_label}")
    print(f"Fine-grain label: {result.fine_grain_label}")
    print(f"Parsed successfully: {result.parsed_successfully}")
    print(f"Response: {result.model_response.response[:100]}...")
    
    assert result.parsed_successfully, "Should parse successfully"
    assert result.binary_label == False, "Should be non-toxic"
    
    print("\n✓ Classify works correctly with API backend")
    return True


def main():
    """Run tests."""
    print("="*70)
    print("BACKWARD COMPATIBILITY TEST")
    print("="*70)
    print("\nThis tests that existing code still works with API backend.")
    print("Note: This will make real API calls (not cached).\n")
    
    try:
        test_api_backend()
        test_classify_api()
        
        print("\n" + "="*70)
        print("✅ ALL BACKWARD COMPATIBILITY TESTS PASSED")
        print("="*70)
        print("\nExisting code continues to work unchanged!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

