# Pre-Commit Checklist

Before committing and transferring to remote machine, verify:

## ✅ Completed Items

- [x] Added `InferenceBackend` enum to `model_predict.py`
- [x] Added `LocalModelCache` singleton class
- [x] Implemented `_get_local_model_response()` with Harmony parsing
- [x] Updated `get_model_response()` to support backend parameter
- [x] Updated `classify()` to accept backend parameter
- [x] Created `compare_backends.py` comparison script
- [x] Created `test_backend_setup.py` verification script
- [x] Created `run_comparison.sh` helper script
- [x] Created comprehensive documentation
- [x] Verified setup with test script (passed ✅)
- [x] Maintained backward compatibility

## 📋 Pre-Commit Verification

Run these commands to verify everything is ready:

```bash
# 1. Verify setup (should pass all tests)
conda run -n gptossexp python -m workspace.test_backend_setup

# Expected output: ✅ ALL TESTS PASSED
```

**Status:** ✅ Passed

## 📦 Files to Commit

### Modified Files
- `workspace/model_predict.py`
- `workspace/classify.py`

### New Files
- `workspace/compare_backends.py`
- `workspace/test_backend_setup.py`
- `workspace/test_api_still_works.py`
- `workspace/run_comparison.sh`
- `workspace/BACKEND_COMPARISON.md`
- `workspace/REMOTE_EXECUTION_GUIDE.md`
- `workspace/CHANGES_SUMMARY.md`
- `workspace/PRE_COMMIT_CHECKLIST.md`

## 🚀 After Commit

On remote machine with GPU:

```bash
# 1. Pull changes
git pull

# 2. Verify setup (no model loading)
conda run -n gptossexp python -m workspace.test_backend_setup

# 3. Run comparison (100 examples)
./workspace/run_comparison.sh 100

# 4. Check results
# Results saved to: workspace/results/backend_comparison/
```

## 📊 Expected Results

- Binary label agreement: ≥95%
- Response text similarity: ≥80%
- Parse success rate: ~100%

## 🔍 If Issues Arise

1. Check `workspace/REMOTE_EXECUTION_GUIDE.md` for troubleshooting
2. Review detailed results in parquet file
3. Try smaller sample size: `./workspace/run_comparison.sh 50`
4. Check memory: `nvidia-smi` or `free -h`

## ✅ Ready to Commit

All checks passed. Ready to commit and transfer to remote machine!

### Suggested Commit Message

```
Add backend comparison feature for API vs LOCAL inference

- Add InferenceBackend enum (API, LOCAL) to model_predict.py
- Add LocalModelCache singleton for model/tokenizer caching
- Add local inference with Harmony format parsing
- Add backend parameter to classify() (defaults to API)
- Create compare_backends.py for validation script
- Create test scripts and comprehensive documentation
- Maintain full backward compatibility

This enables validation that the HF Inference API correctly applies
the Harmony response format by comparing with local transformers
inference on the same examples.

Usage: conda run -n gptossexp python -m workspace.compare_backends --sample-size 100

Files modified:
- workspace/model_predict.py
- workspace/classify.py

Files created:
- workspace/compare_backends.py
- workspace/test_backend_setup.py
- workspace/test_api_still_works.py
- workspace/run_comparison.sh
- workspace/BACKEND_COMPARISON.md
- workspace/REMOTE_EXECUTION_GUIDE.md
- workspace/CHANGES_SUMMARY.md
- workspace/PRE_COMMIT_CHECKLIST.md
```

