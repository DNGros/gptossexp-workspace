"""
Compare API vs LOCAL inference backends on ToxicChat dataset.

This script validates that both backends produce consistent results by:
1. Sampling N examples from ToxicChat
2. Running classification with both API and LOCAL backends
3. Comparing results field-by-field
4. Generating a detailed comparison report

Usage:
    python -m workspace.compare_backends --sample-size 100 --policy toxic_simple
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

app = typer.Typer()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def compare_single_example(
    text: str,
    policy_module,
    model: Model,
    ground_truth: int = None,
) -> dict:
    """
    Run classification on a single example with both backends and compare.
    
    Returns:
        dict with comparison results
    """
    # Run with API backend
    api_result = classify(
        text=text,
        policy_module=policy_module,
        model=model,
        backend=InferenceBackend.API,
    )
    
    # Run with LOCAL backend
    local_result = classify(
        text=text,
        policy_module=policy_module,
        model=model,
        backend=InferenceBackend.LOCAL,
    )
    
    # Compare results
    binary_match = api_result.binary_label == local_result.binary_label
    fine_grain_match = api_result.fine_grain_label == local_result.fine_grain_label
    
    # Calculate text similarities
    response_similarity = calculate_text_similarity(
        api_result.model_response.response,
        local_result.model_response.response
    )
    
    reasoning_similarity = calculate_text_similarity(
        api_result.model_response.reasoning or "",
        local_result.model_response.reasoning or ""
    )
    
    comparison = {
        'text': text,
        'ground_truth': ground_truth,
        
        # API results
        'api_binary_label': api_result.binary_label,
        'api_fine_grain_label': api_result.fine_grain_label,
        'api_parsed_successfully': api_result.parsed_successfully,
        'api_response': api_result.model_response.response,
        'api_reasoning': api_result.model_response.reasoning,
        'api_response_length': len(api_result.model_response.response),
        'api_reasoning_length': len(api_result.model_response.reasoning or ""),
        
        # LOCAL results
        'local_binary_label': local_result.binary_label,
        'local_fine_grain_label': local_result.fine_grain_label,
        'local_parsed_successfully': local_result.parsed_successfully,
        'local_response': local_result.model_response.response,
        'local_reasoning': local_result.model_response.reasoning,
        'local_response_length': len(local_result.model_response.response),
        'local_reasoning_length': len(local_result.model_response.reasoning or ""),
        
        # Comparisons
        'binary_match': binary_match,
        'fine_grain_match': fine_grain_match,
        'response_similarity': response_similarity,
        'reasoning_similarity': reasoning_similarity,
        'both_parsed': api_result.parsed_successfully and local_result.parsed_successfully,
    }
    
    return comparison


def run_backend_comparison(
    sample_size: int = 100,
    policy: str = 'toxic_simple',
    model: str = 'GPT_OSS_20B',
    version: str = 'toxicchat0124',
    split: str = 'test',
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Run backend comparison on ToxicChat dataset.
    
    Args:
        sample_size: Number of examples to compare
        policy: Policy name to use
        model: Model to use
        version: Dataset version
        split: Dataset split
        output_dir: Directory to save results (default: workspace/results/backend_comparison/)
    
    Returns:
        DataFrame with comparison results
    """
    print("="*70)
    print("BACKEND COMPARISON: API vs LOCAL")
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
    
    # Sample examples (stratified by toxicity for balanced coverage)
    print(f"Sampling {sample_size} examples...")
    toxic_examples = df[df['toxicity'] == 1].sample(
        n=min(sample_size // 2, len(df[df['toxicity'] == 1])),
        random_state=42
    )
    non_toxic_examples = df[df['toxicity'] == 0].sample(
        n=min(sample_size // 2, len(df[df['toxicity'] == 0])),
        random_state=42
    )
    sampled_df = pd.concat([toxic_examples, non_toxic_examples]).sample(
        frac=1.0, random_state=42
    ).reset_index(drop=True)
    
    print(f"Sampled: {(sampled_df['toxicity'] == 1).sum()} toxic, "
          f"{(sampled_df['toxicity'] == 0).sum()} non-toxic")
    print()
    
    # Run comparisons
    print("Running comparisons (this will take a while)...")
    print("Note: First LOCAL inference will be slow (loading model)")
    print()
    
    results = []
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Comparing"):
        try:
            comparison = compare_single_example(
                text=row['user_input'],
                policy_module=policy_module,
                model=model_enum,
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
    print_comparison_summary(results_df)
    
    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "backend_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{policy}_{model}_n{sample_size}.parquet"
    results_df.to_parquet(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results_df


def print_comparison_summary(df: pd.DataFrame):
    """Print summary statistics for backend comparison."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    total = len(df)
    
    # Binary label agreement
    binary_agreement = df['binary_match'].sum()
    binary_rate = binary_agreement / total if total > 0 else 0
    print(f"\nBinary Label Agreement: {binary_agreement}/{total} ({binary_rate:.1%})")
    
    if binary_rate < 1.0:
        print(f"  Disagreements: {total - binary_agreement}")
        # Show breakdown of disagreements
        disagreements = df[~df['binary_match']]
        api_true = (disagreements['api_binary_label'] == True).sum()
        local_true = (disagreements['local_binary_label'] == True).sum()
        print(f"    API=True, LOCAL=False: {api_true}")
        print(f"    API=False, LOCAL=True: {local_true}")
    
    # Fine-grain label agreement
    fine_grain_agreement = df['fine_grain_match'].sum()
    fine_grain_rate = fine_grain_agreement / total if total > 0 else 0
    print(f"\nFine-grain Label Agreement: {fine_grain_agreement}/{total} ({fine_grain_rate:.1%})")
    
    # Parse success rates
    api_parsed = df['api_parsed_successfully'].sum()
    local_parsed = df['local_parsed_successfully'].sum()
    both_parsed = df['both_parsed'].sum()
    
    print(f"\nParse Success Rates:")
    print(f"  API: {api_parsed}/{total} ({api_parsed/total:.1%})")
    print(f"  LOCAL: {local_parsed}/{total} ({local_parsed/total:.1%})")
    print(f"  Both: {both_parsed}/{total} ({both_parsed/total:.1%})")
    
    # Text similarities
    avg_response_sim = df['response_similarity'].mean()
    avg_reasoning_sim = df['reasoning_similarity'].mean()
    
    print(f"\nText Similarities (mean):")
    print(f"  Response: {avg_response_sim:.1%}")
    print(f"  Reasoning: {avg_reasoning_sim:.1%}")
    
    # Response lengths
    print(f"\nResponse Lengths (mean):")
    print(f"  API: {df['api_response_length'].mean():.0f} chars")
    print(f"  LOCAL: {df['local_response_length'].mean():.0f} chars")
    
    print(f"\nReasoning Lengths (mean):")
    print(f"  API: {df['api_reasoning_length'].mean():.0f} chars")
    print(f"  LOCAL: {df['local_reasoning_length'].mean():.0f} chars")
    
    # Overall assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if binary_rate >= 0.95:
        print("✅ PASS: Binary labels highly consistent (≥95% agreement)")
    elif binary_rate >= 0.90:
        print("⚠️  CAUTION: Binary labels mostly consistent (≥90% agreement)")
    else:
        print("❌ FAIL: Binary labels inconsistent (<90% agreement)")
    
    if avg_response_sim >= 0.8:
        print("✅ PASS: Response texts highly similar (≥80% similarity)")
    elif avg_response_sim >= 0.6:
        print("⚠️  CAUTION: Response texts moderately similar (≥60% similarity)")
    else:
        print("❌ FAIL: Response texts differ significantly (<60% similarity)")
    
    print("\nConclusion:")
    if binary_rate >= 0.95 and avg_response_sim >= 0.8:
        print("  Both backends produce consistent results. ✓")
        print("  API correctly applies Harmony format. ✓")
    else:
        print("  Backends show significant differences.")
        print("  Review detailed results for debugging.")


@app.command()
def main(
    sample_size: int = typer.Option(100, help="Number of examples to compare"),
    policy: str = typer.Option('toxic_simple', help="Policy name"),
    model: str = typer.Option('GPT_OSS_20B', help="Model to use"),
    version: str = typer.Option('toxicchat0124', help="Dataset version"),
    split: str = typer.Option('test', help="Dataset split"),
):
    """Run backend comparison and generate report."""
    run_backend_comparison(
        sample_size=sample_size,
        policy=policy,
        model=model,
        version=version,
        split=split,
    )


if __name__ == "__main__":
    app()

