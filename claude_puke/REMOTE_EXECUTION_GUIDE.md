# Remote Execution Guide

Quick reference for running the backend comparison on a machine with GPU/sufficient resources.

## Prerequisites

- Machine with 32GB+ RAM or GPU with 24GB+ VRAM
- Conda environment `gptossexp` set up
- Hugging Face token configured (if needed): `huggingface-cli login`

## Quick Start

### 1. Verify Setup (No Model Loading)

```bash
cd /path/to/gptossexp4
conda run -n gptossexp python -m workspace.test_backend_setup
```

Expected output: `✅ ALL TESTS PASSED`

### 2. Run Comparison (100 Examples)

```bash
# Using the helper script
./workspace/run_comparison.sh 100

# Or directly
conda run -n gptossexp python -m workspace.compare_backends --sample-size 100
```

### 3. Check Results

Results are saved to: `workspace/results/backend_comparison/toxic_simple_GPT_OSS_20B_n100.parquet`

The console will show a summary with:
- Binary label agreement rate (target: ≥95%)
- Text similarity scores
- Parse success rates
- Overall pass/fail assessment

## What to Expect

### First Run
- **Model loading**: 1-5 minutes
- **Downloads**: ~40GB if model not cached
- **Per example**: ~1-2 seconds with GPU, ~5-10 seconds with CPU
- **Total time (100 examples)**: 5-20 minutes

### Subsequent Runs
- **No loading**: Model stays in memory
- **Per example**: ~1-2 seconds with GPU
- **Total time (100 examples)**: 2-5 minutes

## Expected Output

```
======================================================================
BACKEND COMPARISON: API vs LOCAL
======================================================================
Policy: toxic_simple
Model: GPT_OSS_20B
Sample size: 100
Dataset: toxicchat0124/test

Loading dataset...
Sampling 100 examples...
Sampled: 50 toxic, 50 non-toxic

Running comparisons (this will take a while)...
Note: First LOCAL inference will be slow (loading model)

Loading model: openai/gpt-oss-20b (this may take a while...)
Model loaded on device: cuda:0

[Progress bar showing 100 examples]

======================================================================
COMPARISON SUMMARY
======================================================================

Binary Label Agreement: 95/100 (95.0%)

Fine-grain Label Agreement: 95/100 (95.0%)

Parse Success Rates:
  API: 100/100 (100.0%)
  LOCAL: 100/100 (100.0%)
  Both: 100/100 (100.0%)

Text Similarities (mean):
  Response: 85.3%
  Reasoning: 82.1%

Response Lengths (mean):
  API: 45 chars
  LOCAL: 43 chars

Reasoning Lengths (mean):
  API: 123 chars
  LOCAL: 118 chars

======================================================================
ASSESSMENT
======================================================================
✅ PASS: Binary labels highly consistent (≥95% agreement)
✅ PASS: Response texts highly similar (≥80% similarity)

Conclusion:
  Both backends produce consistent results. ✓
  API correctly applies Harmony format. ✓

Results saved to: workspace/results/backend_comparison/toxic_simple_GPT_OSS_20B_n100.parquet
```

## Interpreting Results

### ✅ Success (Expected)
- Binary agreement ≥95%
- Response similarity ≥80%
- Both backends parse successfully

**Meaning**: API correctly applies Harmony format, results are trustworthy

### ⚠️ Caution
- Binary agreement 90-95%
- Response similarity 60-80%

**Meaning**: Mostly consistent, but some differences. Review specific disagreements.

### ❌ Failure (Unexpected)
- Binary agreement <90%
- Response similarity <60%

**Meaning**: Significant differences. Investigate detailed results.

## Analyzing Detailed Results

```python
import pandas as pd

# Load results
df = pd.read_parquet('workspace/results/backend_comparison/toxic_simple_GPT_OSS_20B_n100.parquet')

# View all columns
print(df.columns.tolist())

# Check disagreements
disagreements = df[~df['binary_match']]
print(f"\nFound {len(disagreements)} disagreements")

# Look at specific examples
for idx, row in disagreements.head(5).iterrows():
    print(f"\nExample {idx}:")
    print(f"  Text: {row['text'][:100]}...")
    print(f"  Ground truth: {row['ground_truth']}")
    print(f"  API label: {row['api_binary_label']}")
    print(f"  LOCAL label: {row['local_binary_label']}")
    print(f"  API response: {row['api_response'][:100]}...")
    print(f"  LOCAL response: {row['local_response'][:100]}...")

# Check similarity distribution
print("\nResponse similarity distribution:")
print(df['response_similarity'].describe())

# Check if low similarity correlates with disagreements
print("\nAverage similarity for matches vs disagreements:")
print(df.groupby('binary_match')['response_similarity'].mean())
```

## Troubleshooting

### Out of Memory

```bash
# Clear model cache and try again
conda run -n gptossexp python -c "from workspace.model_predict import LocalModelCache; LocalModelCache().clear()"

# Or reduce sample size
./workspace/run_comparison.sh 50
```

### Model Download Fails

```bash
# Login to Hugging Face
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here
```

### Slow Performance

- Check GPU availability: `nvidia-smi`
- Reduce sample size: `--sample-size 50`
- Use CPU if GPU unavailable (will be slower)

## Different Configurations

```bash
# Different policy
conda run -n gptossexp python -m workspace.compare_backends \
  --policy toxic_chat_claude_1 \
  --sample-size 100

# Safeguarded model
conda run -n gptossexp python -m workspace.compare_backends \
  --model GPT_OSS_safeguarded_20B \
  --sample-size 100

# Larger sample
conda run -n gptossexp python -m workspace.compare_backends \
  --sample-size 500
```

## After Running

### If Results Look Good (≥95% agreement)
1. ✅ Commit the results file
2. ✅ Document findings
3. ✅ Use either backend with confidence

### If Results Show Differences
1. 🔍 Inspect detailed results
2. 🔍 Look at specific disagreement examples
3. 🔍 Check if differences are systematic or random
4. 🔍 Consider running with larger sample size
5. 🔍 May need to investigate Harmony format parsing

## Files Modified

- `workspace/model_predict.py` - Added backend support
- `workspace/classify.py` - Added backend parameter
- `workspace/compare_backends.py` - New comparison script
- `workspace/test_backend_setup.py` - Setup verification
- `workspace/run_comparison.sh` - Helper script

## Backward Compatibility

✅ All existing code works unchanged:
- `get_model_response()` defaults to API backend
- `classify()` defaults to API backend
- Existing cache files remain valid
- No changes needed to other scripts

## Questions?

See `workspace/BACKEND_COMPARISON.md` for detailed documentation.

