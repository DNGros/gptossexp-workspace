"""
Compare two inference backends on ToxicChat dataset.

This script compares two backends by:
1. Sampling N examples from ToxicChat
2. Running classification with both backends on the same examples
3. Comparing results field-by-field (agreement metrics)
4. Evaluating ground truth performance for each backend
5. Generating a detailed comparison report

Usage:
    python -m workspace.compare_backends --sample-size 100 --policy toxic_simple \
        --backend1 API --backend2 API_INJECT_HARMONY
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from difflib import SequenceMatcher
import typer

from .classify import classify
from .model_predict import Model, InferenceBackend
from .run_toxic_chat import get_policy_module
from .metrics import calculate_classification_metrics_with_ci, calculate_proportion_with_ci, format_metric_with_ci
from .common.htmlrendering import generate_html_table

app = typer.Typer()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def compare_single_example(
    text: str,
    policy_module,
    model: Model,
    backend1: InferenceBackend,
    backend2: InferenceBackend,
    ground_truth: int = None,
) -> dict:
    """
    Run classification on a single example with both backends and compare.
    
    Args:
        text: Input text to classify
        policy_module: Policy module to use
        model: Model to use
        backend1: First backend to compare
        backend2: Second backend to compare
        ground_truth: Optional ground truth label
    
    Returns:
        dict with comparison results
    """
    # Run with first backend
    result1 = classify(
        text=text,
        policy_module=policy_module,
        model=model,
        backend=backend1,
    )
    
    # Run with second backend
    result2 = classify(
        text=text,
        policy_module=policy_module,
        model=model,
        backend=backend2,
        use_cache=False,
    )
    
    # Compare results
    binary_match = result1.binary_label == result2.binary_label
    fine_grain_match = result1.fine_grain_label == result2.fine_grain_label
    
    # Calculate text similarities
    response_similarity = calculate_text_similarity(
        result1.model_response.response,
        result2.model_response.response
    )
    
    reasoning_similarity = calculate_text_similarity(
        result1.model_response.reasoning or "",
        result2.model_response.reasoning or ""
    )
    
    # Use generic column names that will be renamed later
    comparison = {
        'text': text,
        'ground_truth': ground_truth,
        
        # Backend 1 results
        'backend1_binary_label': result1.binary_label,
        'backend1_fine_grain_label': result1.fine_grain_label,
        'backend1_parsed_successfully': result1.parsed_successfully,
        'backend1_response': result1.model_response.response,
        'backend1_reasoning': result1.model_response.reasoning,
        'backend1_response_length': len(result1.model_response.response),
        'backend1_reasoning_length': len(result1.model_response.reasoning or ""),
        
        # Backend 2 results
        'backend2_binary_label': result2.binary_label,
        'backend2_fine_grain_label': result2.fine_grain_label,
        'backend2_parsed_successfully': result2.parsed_successfully,
        'backend2_response': result2.model_response.response,
        'backend2_reasoning': result2.model_response.reasoning,
        'backend2_response_length': len(result2.model_response.response),
        'backend2_reasoning_length': len(result2.model_response.reasoning or ""),
        
        # Comparisons
        'binary_match': binary_match,
        'fine_grain_match': fine_grain_match,
        'response_similarity': response_similarity,
        'reasoning_similarity': reasoning_similarity,
        'both_parsed': result1.parsed_successfully and result2.parsed_successfully,
    }
    
    return comparison


def run_backend_comparison(
    sample_size: int = 100,
    policy: str = 'toxic_simple',
    model: str = 'GPT_OSS_20B',
    backend1: InferenceBackend = InferenceBackend.API,
    backend2: InferenceBackend = InferenceBackend.API_INJECT_HARMONY,
    version: str = 'toxicchat0124',
    split: str = 'test',
    output_dir: Path = None,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Run backend comparison on ToxicChat dataset.
    
    Args:
        sample_size: Number of examples to compare
        policy: Policy name to use
        model: Model to use
        backend1: First backend to compare
        backend2: Second backend to compare
        version: Dataset version
        split: Dataset split
        output_dir: Directory to save results (default: workspace/results/backend_comparison/)
        n_bootstrap: Number of bootstrap iterations for confidence intervals
    
    Returns:
        DataFrame with comparison results
    """
    backend1_name = backend1.value.upper().replace('_', ' ')
    backend2_name = backend2.value.upper().replace('_', ' ')
    
    print("="*70)
    print(f"BACKEND COMPARISON: {backend1_name} vs {backend2_name}")
    print("="*70)
    print(f"Policy: {policy}")
    print(f"Model: {model}")
    print(f"Sample size: {sample_size}")
    print(f"Dataset: {version}/{split}")
    print()
    
    # Load policy module
    policy_module = get_policy_module(policy)
    model_enum = Model[model]
    
    # Load dataset
    print(f"Loading dataset...")
    dataset = load_dataset("lmsys/toxic-chat", version, split=split)
    df = dataset.to_pandas()
    
    # Sample examples according to natural distribution
    # Use deterministic shuffle so larger samples include smaller ones
    print(f"Sampling {sample_size} examples...")
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sampled_df = df_shuffled.head(min(sample_size, len(df)))
    
    print(f"Sampled: {(sampled_df['toxicity'] == 1).sum()} toxic, "
          f"{(sampled_df['toxicity'] == 0).sum()} non-toxic "
          f"({(sampled_df['toxicity'] == 1).sum() / len(sampled_df):.1%} toxic)")
    print()
    
    # Run comparisons
    print("Running comparisons (this will take a while)...")
    if backend1 in (InferenceBackend.LOCAL, InferenceBackend.LOCAL_INJECT_HARMONY) or \
       backend2 in (InferenceBackend.LOCAL, InferenceBackend.LOCAL_INJECT_HARMONY):
        print("Note: First LOCAL inference will be slow (loading model)")
    print()
    
    results = []
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Comparing"):
        try:
            comparison = compare_single_example(
                text=row['user_input'],
                policy_module=policy_module,
                model=model_enum,
                backend1=backend1,
                backend2=backend2,
                ground_truth=row['toxicity'],
            )
            comparison['conv_id'] = row['conv_id']
            results.append(comparison)
        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            # Continue with other examples
            continue
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print_comparison_summary(results_df, backend1_name, backend2_name, n_bootstrap)
    
    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "backend_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backend1_short = backend1.value
    backend2_short = backend2.value
    
    # Save parquet file
    output_file = output_dir / f"{policy}_{model}_{backend1_short}_vs_{backend2_short}_n{sample_size}.parquet"
    results_df.to_parquet(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Generate and save HTML table
    html_table = generate_backend_comparison_table(
        df=results_df,
        backend1=backend1,
        backend2=backend2,
        policy=policy,
        model=model,
        sample_size=sample_size,
        n_bootstrap=n_bootstrap,
    )
    
    table_dir = Path(__file__).parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_file = table_dir / f"backend_comparison_{policy}_{model}_{backend1_short}_vs_{backend2_short}_n{sample_size}.html"
    table_file.write_text(html_table)
    print(f"HTML table saved to: {table_file}")
    
    return results_df


def print_comparison_summary(df: pd.DataFrame, backend1_name: str, backend2_name: str, n_bootstrap: int = 1000):
    """Print summary statistics for backend comparison.
    
    Args:
        df: DataFrame with comparison results
        backend1_name: Display name for first backend
        backend2_name: Display name for second backend
        n_bootstrap: Number of bootstrap iterations for confidence intervals
    """
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    total = len(df)
    
    # ========== GROUND TRUTH PERFORMANCE ==========
    print("\n" + "="*70)
    print("GROUND TRUTH PERFORMANCE")
    print("="*70)
    
    # Prepare data for metrics calculation
    # Map binary labels to 0/1 for sklearn metrics
    backend1_pred = df['backend1_binary_label'].fillna(False).astype(int).values
    backend2_pred = df['backend2_binary_label'].fillna(False).astype(int).values
    y_true = df['ground_truth'].astype(int).values
    
    # Calculate metrics for backend1
    backend1_metrics = calculate_classification_metrics_with_ci(
        y_true, backend1_pred, n_bootstrap=n_bootstrap
    )
    backend1_parse_rate = calculate_proportion_with_ci(
        df['backend1_parsed_successfully'].values, n_bootstrap=n_bootstrap
    )
    
    # Calculate metrics for backend2
    backend2_metrics = calculate_classification_metrics_with_ci(
        y_true, backend2_pred, n_bootstrap=n_bootstrap
    )
    backend2_parse_rate = calculate_proportion_with_ci(
        df['backend2_parsed_successfully'].values, n_bootstrap=n_bootstrap
    )
    
    # Print backend1 metrics
    print(f"\n{backend1_name}:")
    print(f"  F1:        {format_metric_with_ci(*backend1_metrics['f1'])}")
    print(f"  Precision: {format_metric_with_ci(*backend1_metrics['precision'])}")
    print(f"  Recall:    {format_metric_with_ci(*backend1_metrics['recall'])}")
    print(f"  Parse Rate: {format_metric_with_ci(*backend1_parse_rate, as_percent=True)}")
    
    # Print backend2 metrics with deltas
    print(f"\n{backend2_name}:")
    backend2_f1_mean, backend2_f1_lower, backend2_f1_upper = backend2_metrics['f1']
    backend1_f1_mean = backend1_metrics['f1'][0]
    f1_delta = backend2_f1_mean - backend1_f1_mean
    f1_delta_str = f" [{f1_delta:+.3f}]" if abs(f1_delta) > 0.001 else ""
    
    backend2_prec_mean, backend2_prec_lower, backend2_prec_upper = backend2_metrics['precision']
    backend1_prec_mean = backend1_metrics['precision'][0]
    prec_delta = backend2_prec_mean - backend1_prec_mean
    prec_delta_str = f" [{prec_delta:+.3f}]" if abs(prec_delta) > 0.001 else ""
    
    backend2_rec_mean, backend2_rec_lower, backend2_rec_upper = backend2_metrics['recall']
    backend1_rec_mean = backend1_metrics['recall'][0]
    rec_delta = backend2_rec_mean - backend1_rec_mean
    rec_delta_str = f" [{rec_delta:+.3f}]" if abs(rec_delta) > 0.001 else ""
    
    backend2_parse_mean, backend2_parse_lower, backend2_parse_upper = backend2_parse_rate
    backend1_parse_mean = backend1_parse_rate[0]
    parse_delta = backend2_parse_mean - backend1_parse_mean
    parse_delta_str = f" [{parse_delta*100:+.1f}%]" if abs(parse_delta) > 0.001 else ""
    
    print(f"  F1:        {format_metric_with_ci(*backend2_metrics['f1'])}{f1_delta_str}")
    print(f"  Precision: {format_metric_with_ci(*backend2_metrics['precision'])}{prec_delta_str}")
    print(f"  Recall:    {format_metric_with_ci(*backend2_metrics['recall'])}{rec_delta_str}")
    print(f"  Parse Rate: {format_metric_with_ci(*backend2_parse_rate, as_percent=True)}{parse_delta_str}")
    
    # ========== AGREEMENT BETWEEN BACKENDS ==========
    print("\n" + "="*70)
    print("AGREEMENT BETWEEN BACKENDS")
    print("="*70)
    
    # Binary label agreement
    binary_agreement = df['binary_match'].sum()
    binary_rate = binary_agreement / total if total > 0 else 0
    print(f"\nBinary Label Agreement: {binary_agreement}/{total} ({binary_rate:.1%})")
    
    if binary_rate < 1.0:
        print(f"  Disagreements: {total - binary_agreement}")
        # Show breakdown of disagreements
        disagreements = df[~df['binary_match']]
        backend1_true = (disagreements['backend1_binary_label'] == True).sum()
        backend2_true = (disagreements['backend2_binary_label'] == True).sum()
        print(f"    {backend1_name}=True, {backend2_name}=False: {backend1_true}")
        print(f"    {backend1_name}=False, {backend2_name}=True: {backend2_true}")
    
    # Fine-grain label agreement
    fine_grain_agreement = df['fine_grain_match'].sum()
    fine_grain_rate = fine_grain_agreement / total if total > 0 else 0
    print(f"\nFine-grain Label Agreement: {fine_grain_agreement}/{total} ({fine_grain_rate:.1%})")
    
    # Parse success rates
    backend1_parsed = df['backend1_parsed_successfully'].sum()
    backend2_parsed = df['backend2_parsed_successfully'].sum()
    both_parsed = df['both_parsed'].sum()
    
    print(f"\nParse Success Rates:")
    print(f"  {backend1_name}: {backend1_parsed}/{total} ({backend1_parsed/total:.1%})")
    print(f"  {backend2_name}: {backend2_parsed}/{total} ({backend2_parsed/total:.1%})")
    print(f"  Both: {both_parsed}/{total} ({both_parsed/total:.1%})")
    
    # Text similarities
    avg_response_sim = df['response_similarity'].mean()
    avg_reasoning_sim = df['reasoning_similarity'].mean()
    
    print(f"\nText Similarities (mean):")
    print(f"  Response: {avg_response_sim:.1%}")
    print(f"  Reasoning: {avg_reasoning_sim:.1%}")
    
    # Response lengths
    print(f"\nResponse Lengths (mean):")
    print(f"  {backend1_name}: {df['backend1_response_length'].mean():.0f} chars")
    print(f"  {backend2_name}: {df['backend2_response_length'].mean():.0f} chars")
    
    print(f"\nReasoning Lengths (mean):")
    print(f"  {backend1_name}: {df['backend1_reasoning_length'].mean():.0f} chars")
    print(f"  {backend2_name}: {df['backend2_reasoning_length'].mean():.0f} chars")
    
    # ========== OVERALL ASSESSMENT ==========
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    # Performance comparison
    print("\nPerformance:")
    if abs(f1_delta) < 0.01:
        print(f"  ≈ Similar F1 scores (Δ={f1_delta:+.3f})")
    elif f1_delta > 0:
        print(f"  ✓ {backend2_name} performs better (F1 Δ={f1_delta:+.3f})")
    else:
        print(f"  ✓ {backend1_name} performs better (F1 Δ={-f1_delta:+.3f})")
    
    # Agreement assessment
    print("\nAgreement:")
    if binary_rate >= 0.95:
        print("  ✅ Binary labels highly consistent (≥95% agreement)")
    elif binary_rate >= 0.90:
        print("  ⚠️  Binary labels mostly consistent (≥90% agreement)")
    else:
        print("  ❌ Binary labels inconsistent (<90% agreement)")
    
    if avg_response_sim >= 0.8:
        print("  ✅ Response texts highly similar (≥80% similarity)")
    elif avg_response_sim >= 0.6:
        print("  ⚠️  Response texts moderately similar (≥60% similarity)")
    else:
        print("  ❌ Response texts differ significantly (<60% similarity)")
    
    print("\nConclusion:")
    if binary_rate >= 0.95 and avg_response_sim >= 0.8:
        print("  Both backends produce highly consistent results. ✓")
    elif binary_rate >= 0.90:
        print("  Backends mostly agree, with some differences.")
        print("  Review detailed results for specific disagreements.")
    else:
        print("  Backends show significant differences.")
        print("  Review detailed results for debugging.")


def generate_backend_comparison_table(
    df: pd.DataFrame,
    backend1: InferenceBackend,
    backend2: InferenceBackend,
    policy: str,
    model: str,
    sample_size: int,
    n_bootstrap: int = 1000,
) -> str:
    """Generate HTML table comparing two backends.
    
    Args:
        df: DataFrame with comparison results
        backend1: First backend
        backend2: Second backend
        policy: Policy name
        model: Model name
        sample_size: Number of examples compared
        n_bootstrap: Number of bootstrap iterations for confidence intervals
        
    Returns:
        str: HTML table string
    """
    # Backend display names
    backend1_name = backend1.value.upper().replace('_', ' ')
    backend2_name = backend2.value.upper().replace('_', ' ')
    
    # Prepare data for metrics calculation
    backend1_pred = df['backend1_binary_label'].fillna(False).astype(int).values
    backend2_pred = df['backend2_binary_label'].fillna(False).astype(int).values
    y_true = df['ground_truth'].astype(int).values
    
    # Calculate metrics for both backends
    backend1_metrics = calculate_classification_metrics_with_ci(
        y_true, backend1_pred, n_bootstrap=n_bootstrap
    )
    backend1_parse_rate = calculate_proportion_with_ci(
        df['backend1_parsed_successfully'].values, n_bootstrap=n_bootstrap
    )
    
    backend2_metrics = calculate_classification_metrics_with_ci(
        y_true, backend2_pred, n_bootstrap=n_bootstrap
    )
    backend2_parse_rate = calculate_proportion_with_ci(
        df['backend2_parsed_successfully'].values, n_bootstrap=n_bootstrap
    )
    
    # Format metrics
    def format_metrics_dict(metrics, parse_rate):
        return {
            'F1': format_metric_with_ci(*metrics['f1']),
            'Precision': format_metric_with_ci(*metrics['precision']),
            'Recall': format_metric_with_ci(*metrics['recall']),
            'Parse Rate': format_metric_with_ci(*parse_rate, as_percent=True),
        }
    
    # Create table rows
    rows = [
        {
            'Backend': backend1_name,
            'Metrics': format_metrics_dict(backend1_metrics, backend1_parse_rate),
        },
        {
            'Backend': backend2_name,
            'Metrics': format_metrics_dict(backend2_metrics, backend2_parse_rate),
        }
    ]
    
    # Generate HTML table
    return generate_html_table(rows=rows)


@app.command()
def main(
    sample_size: int = typer.Option(100, help="Number of examples to compare"),
    policy: str = typer.Option('toxic_simple', help="Policy name"),
    model: str = typer.Option('GPT_OSS_safeguarded_20B', help="Model to use"),
    backend1: str = typer.Option('API', help="First backend (API, API_INJECT_HARMONY, LOCAL, LOCAL_INJECT_HARMONY)"),
    backend2: str = typer.Option('API_INJECT_HARMONY', help="Second backend (API, API_INJECT_HARMONY, LOCAL, LOCAL_INJECT_HARMONY)"),
    version: str = typer.Option('toxicchat0124', help="Dataset version"),
    split: str = typer.Option('test', help="Dataset split"),
    n_bootstrap: int = typer.Option(1000, help="Number of bootstrap iterations for confidence intervals"),
):
    """Run backend comparison and generate report.
    
    Example:
        python -m workspace.compare_backends --backend1 API --backend2 API_INJECT_HARMONY --sample-size 100
    """
    # Convert string backend names to enum
    backend1_enum = InferenceBackend[backend1]
    backend2_enum = InferenceBackend[backend2]
    
    run_backend_comparison(
        sample_size=sample_size,
        policy=policy,
        model=model,
        backend1=backend1_enum,
        backend2=backend2_enum,
        version=version,
        split=split,
        n_bootstrap=n_bootstrap,
    )


if __name__ == "__main__":
    app()

