# Backend Comparison: API vs LOCAL Inference

This document describes the backend comparison feature added to validate that the Hugging Face Inference API and local transformers inference produce consistent results.

## Overview

The codebase now supports two inference backends:
- **API** (default): Hugging Face Inference API - fast, no local resources needed
- **LOCAL**: Local transformers with `apply_chat_template` - slower, requires GPU/CPU resources

Both backends should produce identical or highly similar results if the API correctly applies the Harmony response format.

## Changes Made

### 1. `model_predict.py`
- Added `InferenceBackend` enum with `API` and `LOCAL` values
- Added `LocalModelCache` singleton class to cache models/tokenizers
- Added `_get_local_model_response()` function for local inference
- Added `_parse_harmony_channels()` to extract reasoning and content from Harmony format
- Updated `get_model_response()` to accept optional `backend` parameter (defaults to `API`)
- Updated cache key generation to include backend type

### 2. `classify.py`
- Added `backend` parameter to `classify()` function (defaults to `API`)
- Passes backend through to `get_model_response()`

### 3. `compare_backends.py` (new)
- Main comparison script
- Samples examples from ToxicChat dataset
- Runs classification with both backends
- Compares results and generates detailed report

### 4. `test_backend_setup.py` (new)
- Verification script to test setup without running models
- Tests imports, enums, function signatures, and backward compatibility

## Backward Compatibility

✅ **All existing code continues to work unchanged**
- `backend` parameter is optional and defaults to `API`
- Existing cache files remain valid (different cache keys for different backends)
- No changes needed to existing scripts

## Usage

### Running the Comparison

On a machine with sufficient resources (GPU recommended):

```bash
# Basic usage (100 examples, toxic_simple policy)
conda run -n gptossexp python -m workspace.compare_backends

# Custom sample size
conda run -n gptossexp python -m workspace.compare_backends --sample-size 200

# Different policy
conda run -n gptossexp python -m workspace.compare_backends --policy toxic_chat_claude_1

# Full options
conda run -n gptossexp python -m workspace.compare_backends \
  --sample-size 100 \
  --policy toxic_simple \
  --model GPT_OSS_20B \
  --version toxicchat0124 \
  --split test
```

### Using LOCAL Backend in Your Code

```python
from workspace.classify import classify
from workspace.model_predict import Model, InferenceBackend
from workspace.policies import toxic_simple

# Use LOCAL backend
result = classify(
    text="Your text here",
    policy_module=toxic_simple,
    model=Model.GPT_OSS_20B,
    backend=InferenceBackend.LOCAL,  # <-- Specify LOCAL
)

# Or use API backend (default)
result = classify(
    text="Your text here",
    policy_module=toxic_simple,
    model=Model.GPT_OSS_20B,
    # backend defaults to API
)
```

## Expected Results

### If Backends Match (≥95% agreement):
✅ Validates that API correctly applies Harmony format  
✅ Provides confidence in existing results  
✅ Enables local inference for debugging/development  

### If Backends Differ Significantly:
⚠️ Indicates potential format mismatch  
🔍 Provides detailed comparison for debugging  
📊 Shows where differences occur (parsing, generation, etc.)  

## Output

The comparison script generates:

1. **Console Report**:
   - Binary label agreement rate
   - Fine-grain label agreement rate
   - Parse success rates for both backends
   - Text similarity scores
   - Overall pass/fail assessment

2. **Parquet File** (saved to `workspace/results/backend_comparison/`):
   - Detailed results for each example
   - Both API and LOCAL outputs
   - Similarity scores
   - Ground truth labels

## Performance Notes

### First Run
- **Slow**: Model loading takes 1-5 minutes depending on hardware
- Downloads model weights if not cached (~40GB for gpt-oss-20b)

### Subsequent Runs
- **Fast**: Model stays in memory (singleton cache)
- Only inference time, no loading overhead

### Memory Requirements
- **gpt-oss-20b**: ~40GB disk space, ~20GB RAM/VRAM
- Recommend: 32GB+ RAM or GPU with 24GB+ VRAM

## Troubleshooting

### Out of Memory
```python
from workspace.model_predict import LocalModelCache

# Clear cached models to free memory
cache = LocalModelCache()
cache.clear()
```

### Model Not Found
Ensure you have access to the model on Hugging Face:
```bash
huggingface-cli login
```

### Import Errors
The transformers/torch imports are lazy-loaded. If you see import warnings in your IDE, they're expected and won't cause runtime errors.

## Testing Without Running Models

To verify setup is correct without loading models:

```bash
conda run -n gptossexp python -m workspace.test_backend_setup
```

This runs quick tests to ensure:
- All imports work
- Enums are defined correctly
- Function signatures are correct
- Backward compatibility is maintained

## File Locations

- **Code**: `workspace/model_predict.py`, `workspace/classify.py`
- **Comparison Script**: `workspace/compare_backends.py`
- **Test Script**: `workspace/test_backend_setup.py`
- **Results**: `workspace/results/backend_comparison/`
- **Cache**: `.cache/model_responses/` (API and LOCAL have separate cache keys)

## Next Steps

After running the comparison:

1. Review the console report
2. Check agreement rates (target: ≥95%)
3. Inspect detailed results in parquet file if needed
4. If backends match, you can confidently use either for your work
5. If backends differ, investigate specific examples where they disagree

