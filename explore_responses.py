"""
Script to explore response and reasoning length distributions from ToxicChat evaluation results.

This script:
1. Loads parquet result files from the results/ directory
2. Analyzes the length distribution of model_reasoning and raw_model_response
3. Provides statistics and visualizations
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import typer
from collections import Counter
import numpy as np

app = typer.Typer()


def analyze_length_distribution(text_series: pd.Series, name: str):
    """Analyze and print length distribution statistics."""
    # Convert to string and handle NaN/None values
    text_series = text_series.astype(str)
    text_series = text_series.replace('nan', '')
    text_series = text_series.replace('None', '')
    
    # Calculate lengths (in characters)
    lengths = text_series.str.len()
    
    print(f"\n{'='*60}")
    print(f"{name} Length Distribution (characters)")
    print(f"{'='*60}")
    print(f"Total examples: {len(lengths)}")
    print(f"Empty/missing: {(lengths == 0).sum()} ({(lengths == 0).sum() / len(lengths) * 100:.1f}%)")
    print(f"Non-empty: {(lengths > 0).sum()} ({(lengths > 0).sum() / len(lengths) * 100:.1f}%)")
    
    if lengths.sum() > 0:
        print(f"\nBasic Statistics:")
        print(f"  Mean: {lengths.mean():.1f} characters")
        print(f"  Median: {lengths.median():.1f} characters")
        print(f"  Std Dev: {lengths.std():.1f} characters")
        print(f"  Min: {lengths.min()} characters")
        print(f"  Max: {lengths.max()} characters")
        print(f"  Total characters: {lengths.sum():,}")
        
        print(f"\nPercentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"  {p}th percentile: {np.percentile(lengths, p):.1f} characters")
        
        # Show some examples of different lengths
        print(f"\nExamples:")
        if (lengths > 0).any():
            non_zero_lengths = lengths[lengths > 0]
            print(f"  Shortest non-empty ({non_zero_lengths.min()} chars):")
            idx = lengths[lengths == non_zero_lengths.min()].index[0]
            sample = text_series.iloc[idx]
            print(f"    {repr(sample[:100])}{'...' if len(sample) > 100 else ''}")
            
            print(f"  Median length ({non_zero_lengths.median():.0f} chars):")
            median_val = non_zero_lengths.median()
            closest_idx = lengths[lengths > 0].sub(median_val).abs().idxmin()
            sample = text_series.iloc[closest_idx]
            print(f"    {repr(sample[:200])}{'...' if len(sample) > 200 else ''}")
            
            print(f"  Longest ({non_zero_lengths.max()} chars):")
            idx = lengths.idxmax()
            sample = text_series.iloc[idx]
            print(f"    {repr(sample[:200])}{'...' if len(sample) > 200 else ''}")
    
    # Calculate word counts for non-empty responses
    word_counts = text_series[text_series.str.len() > 0].str.split().str.len()
    if len(word_counts) > 0:
        print(f"\nWord Count Statistics (non-empty only):")
        print(f"  Mean: {word_counts.mean():.1f} words")
        print(f"  Median: {word_counts.median():.1f} words")
        print(f"  Min: {word_counts.min()} words")
        print(f"  Max: {word_counts.max()} words")
    
    return lengths


def find_result_files(results_dir: Path, pattern: Optional[str] = None) -> List[Path]:
    """Find all parquet result files in the results directory."""
    parquet_files = []
    
    if pattern:
        # Look for specific pattern match
        for file in results_dir.rglob("*.parquet"):
            if pattern in file.name:
                parquet_files.append(file)
    else:
        # Find all parquet files
        parquet_files = list(results_dir.rglob("*.parquet"))
    
    return sorted(parquet_files)


@app.command()
def explore(
    results_dir: Optional[Path] = typer.Option(None, help='Results directory (default: workspace/results)'),
    file_pattern: Optional[str] = typer.Option(None, help='Pattern to match in filename (e.g., "toxic_chat_toxic_chat_claude_1")'),
    file_path: Optional[Path] = typer.Option(None, help='Specific parquet file to analyze'),
):
    """Explore response and reasoning length distributions from result files."""
    
    # Determine which files to analyze
    if file_path:
        if not file_path.exists():
            raise typer.BadParameter(f"File not found: {file_path}")
        files_to_analyze = [file_path]
    else:
        if results_dir is None:
            workspace_dir = Path(__file__).parent
            results_dir = workspace_dir / "results"
        else:
            results_dir = Path(results_dir)
        
        if not results_dir.exists():
            raise typer.BadParameter(f"Results directory not found: {results_dir}")
        
        files_to_analyze = find_result_files(results_dir, file_pattern)
        
        if not files_to_analyze:
            raise typer.BadParameter(f"No parquet files found in {results_dir}" + 
                                    (f" matching pattern '{file_pattern}'" if file_pattern else ""))
    
    print(f"Found {len(files_to_analyze)} file(s) to analyze")
    
    # Analyze each file
    all_reasoning_lengths = []
    all_response_lengths = []
    
    for file_path in files_to_analyze:
        print(f"\n{'#'*60}")
        print(f"Analyzing: {file_path}")
        print(f"{'#'*60}")
        
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ['model_reasoning', 'raw_model_response']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, skipping...")
            continue
        
        # Analyze reasoning lengths
        reasoning_lengths = analyze_length_distribution(
            df['model_reasoning'], 
            "Model Reasoning"
        )
        all_reasoning_lengths.extend(reasoning_lengths.tolist())
        
        # Analyze response lengths
        response_lengths = analyze_length_distribution(
            df['raw_model_response'], 
            "Raw Model Response"
        )
        all_response_lengths.extend(response_lengths.tolist())
        
        # Show comparison
        print(f"\n{'='*60}")
        print("Comparison")
        print(f"{'='*60}")
        if len(reasoning_lengths) > 0 and len(response_lengths) > 0:
            reasoning_non_zero = reasoning_lengths[reasoning_lengths > 0]
            response_non_zero = response_lengths[response_lengths > 0]
            
            if len(reasoning_non_zero) > 0 and len(response_non_zero) > 0:
                print(f"Reasoning avg length: {reasoning_non_zero.mean():.1f} chars")
                print(f"Response avg length: {response_non_zero.mean():.1f} chars")
                print(f"Ratio (reasoning/response): {reasoning_non_zero.mean() / response_non_zero.mean():.2f}")
    
    # Overall summary if multiple files
    if len(files_to_analyze) > 1:
        print(f"\n{'#'*60}")
        print("OVERALL SUMMARY (across all files)")
        print(f"{'#'*60}")
        
        all_reasoning = pd.Series(all_reasoning_lengths)
        all_response = pd.Series(all_response_lengths)
        
        print(f"\nReasoning (aggregated):")
        print(f"  Mean: {all_reasoning.mean():.1f} chars")
        print(f"  Median: {all_reasoning.median():.1f} chars")
        print(f"  Non-empty: {(all_reasoning > 0).sum()}/{len(all_reasoning)}")
        
        print(f"\nResponse (aggregated):")
        print(f"  Mean: {all_response.mean():.1f} chars")
        print(f"  Median: {all_response.median():.1f} chars")
        print(f"  Non-empty: {(all_response > 0).sum()}/{len(all_response)}")


if __name__ == "__main__":
    app()

