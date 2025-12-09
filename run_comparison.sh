#!/bin/bash
# Helper script to run backend comparison on remote machine
# Usage: ./workspace/run_comparison.sh [sample_size]

set -e  # Exit on error

SAMPLE_SIZE=${1:-100}
POLICY=${2:-toxic_simple}
MODEL=${3:-GPT_OSS_20B}

echo "========================================================================"
echo "BACKEND COMPARISON: API vs LOCAL"
echo "========================================================================"
echo "Sample size: $SAMPLE_SIZE"
echo "Policy: $POLICY"
echo "Model: $MODEL"
echo ""
echo "This will:"
echo "  1. Load the model locally (first run: ~5 min)"
echo "  2. Run $SAMPLE_SIZE examples through both API and LOCAL backends"
echo "  3. Compare results and generate report"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "Starting comparison..."
echo ""

# Run the comparison
conda run -n gptossexp python -m workspace.compare_backends \
  --sample-size "$SAMPLE_SIZE" \
  --policy "$POLICY" \
  --model "$MODEL"

echo ""
echo "========================================================================"
echo "COMPARISON COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: workspace/results/backend_comparison/"
echo ""
echo "To view results:"
echo "  import pandas as pd"
echo "  df = pd.read_parquet('workspace/results/backend_comparison/${POLICY}_${MODEL}_n${SAMPLE_SIZE}.parquet')"
echo "  df.head()"

