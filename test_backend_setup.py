"""
Quick test to verify backend setup is correct (without running models).

This tests:
1. Imports work correctly
2. Enums are defined
3. Functions have correct signatures
4. No syntax errors
"""

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    from workspace.model_predict import (
        Model,
        InferenceBackend,
        ModelResponse,
        get_model_response,
        LocalModelCache,
    )
    from workspace.classify import classify, ClassificationResult
    from workspace.compare_backends import (
        run_backend_comparison,
        compare_single_example,
    )
    
    print("✓ All imports successful")
    return True


def test_enums():
    """Test that enums are defined correctly."""
    print("\nTesting enums...")
    
    from workspace.model_predict import Model, InferenceBackend
    
    # Test Model enum
    assert hasattr(Model, 'GPT_OSS_20B')
    assert hasattr(Model, 'GPT_OSS_safeguarded_20B')
    print(f"  Model.GPT_OSS_20B = {Model.GPT_OSS_20B}")
    
    # Test InferenceBackend enum
    assert hasattr(InferenceBackend, 'API')
    assert hasattr(InferenceBackend, 'LOCAL')
    print(f"  InferenceBackend.API = {InferenceBackend.API}")
    print(f"  InferenceBackend.LOCAL = {InferenceBackend.LOCAL}")
    
    print("✓ Enums defined correctly")
    return True


def test_function_signatures():
    """Test that functions have correct signatures."""
    print("\nTesting function signatures...")
    
    import inspect
    from workspace.model_predict import get_model_response
    from workspace.classify import classify
    
    # Check get_model_response signature
    sig = inspect.signature(get_model_response)
    params = list(sig.parameters.keys())
    assert 'backend' in params, "get_model_response should have 'backend' parameter"
    
    # Check default value
    backend_param = sig.parameters['backend']
    assert backend_param.default is not inspect.Parameter.empty, "backend should have default value"
    print(f"  get_model_response has backend parameter with default: {backend_param.default}")
    
    # Check classify signature
    sig = inspect.signature(classify)
    params = list(sig.parameters.keys())
    assert 'backend' in params, "classify should have 'backend' parameter"
    
    backend_param = sig.parameters['backend']
    assert backend_param.default is not inspect.Parameter.empty, "backend should have default value"
    print(f"  classify has backend parameter with default: {backend_param.default}")
    
    print("✓ Function signatures correct")
    return True


def test_cache_singleton():
    """Test that LocalModelCache is a singleton."""
    print("\nTesting LocalModelCache singleton...")
    
    from workspace.model_predict import LocalModelCache
    
    cache1 = LocalModelCache()
    cache2 = LocalModelCache()
    
    assert cache1 is cache2, "LocalModelCache should be a singleton"
    print("✓ LocalModelCache is a singleton")
    return True


def test_backward_compatibility():
    """Test that old code still works (without backend parameter)."""
    print("\nTesting backward compatibility...")
    
    from workspace.model_predict import get_model_response, Model, InferenceBackend
    import inspect
    
    # Verify that backend parameter is optional
    sig = inspect.signature(get_model_response)
    backend_param = sig.parameters['backend']
    
    # Check it defaults to API
    assert backend_param.default == InferenceBackend.API, "backend should default to API"
    print("✓ Backend defaults to API (backward compatible)")
    
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("BACKEND SETUP VERIFICATION")
    print("="*70)
    
    tests = [
        test_imports,
        test_enums,
        test_function_signatures,
        test_cache_singleton,
        test_backward_compatibility,
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nSetup is ready for remote execution!")
        print("\nTo run the comparison:")
        print("  conda run -n gptossexp python -m workspace.compare_backends --sample-size 100")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix errors before running on remote machine.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

