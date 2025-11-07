"""
Script to run toxicity classification on the ToxicChat dataset.

This script:
1. Loads the ToxicChat dataset from Hugging Face
2. Runs classification on user_input using specified policy and model
3. Compares results with ground truth labels
4. Saves results to parquet format in results/ folder with descriptive filename

Dataset Schema:
===============
The ToxicChat dataset contains the following fields:

Input Fields (from dataset):
- conv_id (str): Unique conversation identifier (64 chars)
- user_input (str): The user message to classify (12-1540 chars)
- model_output (str): The AI model's response (1-4120 chars)
- human_annotation (bool): Whether this example was human-annotated
- toxicity (int): Ground truth toxicity label (0=non-toxic, 1=toxic)
- jailbreaking (int): Ground truth jailbreaking label (0=no, 1=yes)
- openai_moderation (str): JSON string with OpenAI moderation API scores

Output Fields (added by classification):
- policy_name (str): Name of the policy used for classification
- model_name (str): Name of the model used for classification
- predicted_toxic (bool): Our model's binary prediction (True=toxic, False=non-toxic)
- predicted_label (str): Fine-grain label ("0" or "1")
- parsed_successfully (bool): Whether the parser successfully extracted meaningful information from the model response
- raw_model_response (str): Full text response from classification model
- model_reasoning (str): Reasoning from classification model (if available)
- model_used (str): Model enum value used
- matches_ground_truth (bool): Whether prediction matches toxicity ground truth
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Literal, Optional
import typer
import importlib
import numpy as np
from enum import Enum
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from .classify import classify
from .model_predict import Model, InferenceBackend, DEFAULT_BATCH_SIZE

app = typer.Typer()


class SamplingStrategy(str, Enum):
    """Strategy for sampling from imbalanced dataset."""
    NATURAL = "natural"      # Sample proportional to true distribution
    BALANCED = "balanced"    # 50/50 split between positive and negative
    NEYMAN = "neyman"        # Sample proportional to sqrt(class_proportion)


def calculate_target_counts(
    total_available_pos: int,
    total_available_neg: int,
    sample_size: int,
    strategy: SamplingStrategy,
    true_pos_rate: float
) -> tuple[int, int]:
    """
    Calculate how many positive and negative examples to sample.
    
    Args:
        total_available_pos: Total positive examples available in dataset
        total_available_neg: Total negative examples available in dataset
        sample_size: Desired total sample size
        strategy: Sampling strategy to use
        true_pos_rate: True positive rate in the population
        
    Returns:
        tuple: (n_pos_to_sample, n_neg_to_sample)
        
    Note: Honors sample_size by filling with other class if insufficient samples.
    """
    if strategy == SamplingStrategy.NATURAL:
        target_pos = int(sample_size * true_pos_rate)
        target_neg = sample_size - target_pos
        
    elif strategy == SamplingStrategy.BALANCED:
        target_pos = sample_size // 2
        target_neg = sample_size - target_pos
        
    elif strategy == SamplingStrategy.NEYMAN:
        # Neyman allocation: sample proportional to sqrt(class_proportion)
        sqrt_pos = np.sqrt(true_pos_rate)
        sqrt_neg = np.sqrt(1 - true_pos_rate)
        ratio_pos = sqrt_pos / (sqrt_pos + sqrt_neg)
        target_pos = int(sample_size * ratio_pos)
        target_neg = sample_size - target_pos
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Handle insufficient samples
    actual_pos = min(target_pos, total_available_pos)
    actual_neg = min(target_neg, total_available_neg)
    
    # If we couldn't get enough of one class, try to fill with the other
    if actual_pos < target_pos:
        # Not enough positives, take more negatives
        actual_neg = min(sample_size - actual_pos, total_available_neg)
    elif actual_neg < target_neg:
        # Not enough negatives (unlikely!), take more positives
        actual_pos = min(sample_size - actual_neg, total_available_pos)
    
    return actual_pos, actual_neg


def stratified_sample(
    df: pd.DataFrame,
    label_column: str,
    n_positive: int,
    n_negative: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample stratified by label with specified counts for each class.
    
    Args:
        df: DataFrame to sample from
        label_column: Name of the binary label column
        n_positive: Number of positive examples to sample
        n_negative: Number of negative examples to sample
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled examples
    """
    df_pos = df[df[label_column] == 1]
    df_neg = df[df[label_column] == 0]
    
    # Sample from each class
    sampled_pos = df_pos.sample(n=min(n_positive, len(df_pos)), random_state=random_state)
    sampled_neg = df_neg.sample(n=min(n_negative, len(df_neg)), random_state=random_state)
    
    # Combine and shuffle
    sampled_df = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    return sampled_df


def calculate_sample_weights(
    labels: np.ndarray,
    true_pos_rate: float,
    strategy: SamplingStrategy
) -> np.ndarray:
    """
    Calculate sample weights for reweighting to population distribution.
    
    Args:
        labels: Binary labels (0 or 1)
        true_pos_rate: True positive rate in the population
        strategy: Sampling strategy used
        
    Returns:
        Array of sample weights
        
    Note: For natural sampling, all weights are 1.0 (no reweighting needed).
    """
    if strategy == SamplingStrategy.NATURAL:
        # No reweighting needed for natural sampling
        return np.ones(len(labels))
    
    # Calculate actual sampling proportions
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    total = len(labels)
    
    actual_pos_rate = n_pos / total
    actual_neg_rate = n_neg / total
    
    # Weight = true_proportion / sample_proportion
    pos_weight = true_pos_rate / actual_pos_rate if actual_pos_rate > 0 else 1.0
    neg_weight = (1 - true_pos_rate) / actual_neg_rate if actual_neg_rate > 0 else 1.0
    
    # Assign weights based on labels
    weights = np.where(labels == 1, pos_weight, neg_weight)
    
    return weights


def get_policy_module(policy_name: str):
    """Dynamically import a policy module by name."""
    try:
        # Use relative import like other files in the workspace
        policy_module = importlib.import_module(f".policies.{policy_name}", package=__package__)
        return policy_module
    except (ImportError, AttributeError) as e:
        # Fallback: try absolute import
        try:
            policy_module = importlib.import_module(f"workspace.policies.{policy_name}")
            return policy_module
        except ImportError:
            raise typer.BadParameter(f"Policy '{policy_name}' not found. Available: spam, toxic_chat_claude_1") from e


def get_cache_file_path(
    version: str,
    model: str,
    policy: str,
    split: str,
    sample_size: int,
    workspace_dir: Optional[Path] = None,
    sampling_strategy: SamplingStrategy = SamplingStrategy.NATURAL
) -> Path:
    """Generate the expected cache file path for a given configuration."""
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent
    results_dir = workspace_dir / "results" / version / model
    
    # Add sampling strategy suffix for non-natural sampling
    strategy_suffix = "" if sampling_strategy == SamplingStrategy.NATURAL else f"_{sampling_strategy.value}"
    base_filename = f"toxic_chat_{policy}_{split}_n{sample_size}{strategy_suffix}.parquet"
    
    return results_dir / base_filename


def print_summary_stats(results_df: pd.DataFrame):
    """Print summary statistics and confusion matrix."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total examples processed: {len(results_df)}")
    
    # Show sampling information if weights are present
    if 'sample_weight' in results_df.columns and 'sampling_strategy' in results_df.columns:
        strategy = results_df['sampling_strategy'].iloc[0]
        if strategy != SamplingStrategy.NATURAL.value:
            weights = results_df['sample_weight']
            print(f"\nSampling Strategy: {strategy}")
            print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
            print(f"  Unique weights: {weights.nunique()}")
    
    if 'parsed_successfully' in results_df.columns:
        successful_parses = results_df['parsed_successfully'].sum()
        total_parses = results_df['parsed_successfully'].notna().sum()
        if total_parses > 0:
            print(f"\nParsing Success Rate:")
            print(f"  Successfully parsed: {successful_parses}/{total_parses} ({successful_parses/total_parses:.2%})")
            print(f"  Failed to parse: {total_parses - successful_parses}/{total_parses} ({(total_parses - successful_parses)/total_parses:.2%})")
    
    if results_df['predicted_toxic'].notna().any():
        print(f"\nGround Truth Toxicity Distribution:")
        print(f"  Toxic (1): {(results_df['toxicity'] == 1).sum()}")
        print(f"  Non-toxic (0): {(results_df['toxicity'] == 0).sum()}")
        
        print(f"\nPredicted Toxicity Distribution:")
        print(f"  Toxic (True): {results_df['predicted_toxic'].sum()}")
        print(f"  Non-toxic (False): {(~results_df['predicted_toxic']).sum()}")
        
        print(f"\nAccuracy:")
        matches = results_df['matches_ground_truth'].sum()
        total = results_df['matches_ground_truth'].notna().sum()
        accuracy = matches / total if total > 0 else 0
        print(f"  Correct: {matches}/{total} ({accuracy:.2%})")
        
        # Convert to binary labels for sklearn (0/1)
        y_true = results_df['toxicity'].astype(int)
        y_pred = results_df['predicted_toxic'].astype(int)
        
        # Confusion matrix (explicitly specify labels to ensure 2x2 matrix)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Negatives (TN): {tn}")
        
        # Calculate precision, recall, and F1 score using sklearn
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")


def _create_result_row(row, classification_result, policy: str, model: str):
    """Helper to create result row dictionary from classification result.

    Args:
        row: DataFrame row (either from iterrows or itertuples)
        classification_result: ClassificationResult object
        policy: Policy name
        model: Model name

    Returns:
        dict: Result row dictionary
    """
    # Handle both iterrows and itertuples access patterns
    def get_field(field_name):
        return getattr(row, field_name, None) or row.get(field_name) if hasattr(row, 'get') else getattr(row, field_name)

    return {
        'conv_id': get_field('conv_id'),
        'user_input': get_field('user_input'),
        'model_output': get_field('model_output'),
        'human_annotation': get_field('human_annotation'),
        'toxicity': get_field('toxicity'),
        'jailbreaking': get_field('jailbreaking'),
        'openai_moderation': get_field('openai_moderation'),
        'policy_name': policy,
        'model_name': model,
        'predicted_toxic': classification_result.binary_label,
        'predicted_label': classification_result.fine_grain_label,
        'parsed_successfully': classification_result.parsed_successfully,
        'raw_model_response': classification_result.model_response.response,
        'model_reasoning': classification_result.model_response.reasoning,
        'model_used': classification_result.model_response.model,
        'matches_ground_truth': (
            classification_result.binary_label == (get_field('toxicity') == 1)
        ),
    }


def run_toxic_chat_evaluation(
    sample_size: int = 20,
    version: Literal['toxicchat0124', 'toxicchat1123'] = 'toxicchat0124',
    split: Literal['train', 'test'] = 'test',
    policy: str = 'toxic_chat_claude_1',
    model: Literal['GPT_OSS_20B', 'GPT_OSS_safeguarded_20B'] = 'GPT_OSS_20B',
    output: Optional[Path] = None,
    use_cache: bool = True,
    sampling_strategy: SamplingStrategy = SamplingStrategy.NATURAL,
    backend: InferenceBackend = InferenceBackend.API,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Run toxicity classification on ToxicChat dataset.

    Args:
        sample_size: Number of examples to sample
        version: Dataset version
        split: Dataset split (train/test)
        policy: Policy name to use
        model: Model to use
        output: Optional output path (auto-generated if None)
        use_cache: Whether to load from cache if available
        sampling_strategy: Strategy for sampling (natural/balanced/neyman)
        backend: Inference backend (API or LOCAL)
        batch_size: Batch size for LOCAL backend (only used if backend=LOCAL)

    Note: The true positive rate is automatically inferred from the full dataset
          before sampling, ensuring accurate weight calculation.
    """
    # Check cache first if use_cache is True and output is not explicitly provided
    if use_cache and output is None:
        workspace_dir = Path(__file__).parent
        cache_file = get_cache_file_path(version, model, policy, split, sample_size, workspace_dir, sampling_strategy)
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            results_df = pd.read_parquet(cache_file)
            print_summary_stats(results_df)
            print(f"\nResults DataFrame shape: {results_df.shape}")
            print(f"Columns: {list(results_df.columns)}")
            return results_df
    
    # Load policy module
    policy_module = get_policy_module(policy)
    
    # Get model enum
    model_enum = Model[model]
    
    print(f"Using policy: {policy}")
    print(f"Using model: {model} ({model_enum})")
    print(f"Using sampling strategy: {sampling_strategy.value}")
    
    dataset = load_dataset("lmsys/toxic-chat", version, split=split)
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    df = dataset.to_pandas()
    
    # Infer true positive rate from full dataset
    true_positive_rate = (df['toxicity'] == 1).sum() / len(df)
    print(f"True positive rate in dataset: {true_positive_rate:.4f} ({(df['toxicity'] == 1).sum()}/{len(df)})")
    
    # Apply sampling strategy
    if sampling_strategy == SamplingStrategy.NATURAL:
        # Natural sampling: random sample from full dataset
        print(f"Sampling {sample_size} examples (natural sampling)...")
        if len(df) > sample_size:
            df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            df = df_shuffled.head(sample_size).reset_index(drop=True)
    else:
        # Stratified sampling: sample by class
        total_pos = (df['toxicity'] == 1).sum()
        total_neg = (df['toxicity'] == 0).sum()
        
        n_pos, n_neg = calculate_target_counts(
            total_available_pos=total_pos,
            total_available_neg=total_neg,
            sample_size=sample_size,
            strategy=sampling_strategy,
            true_pos_rate=true_positive_rate
        )
        
        print(f"Sampling {sample_size} examples ({sampling_strategy.value} sampling)...")
        print(f"  Target: {n_pos} positive, {n_neg} negative")
        print(f"  Available: {total_pos} positive, {total_neg} negative")
        
        df = stratified_sample(
            df=df,
            label_column='toxicity',
            n_positive=n_pos,
            n_negative=n_neg,
            random_state=42
        )
        
        print(f"  Sampled: {(df['toxicity'] == 1).sum()} positive, {(df['toxicity'] == 0).sum()} negative")
    
    print(f"Processing {len(df)} examples...")
    print(f"Using backend: {backend.value}")
    if backend == InferenceBackend.LOCAL:
        print(f"Batch size: {batch_size}")

    # Process all examples at once - let classify() and pipeline handle batching internally
    classification_results = classify(
        text=df['user_input'].tolist(),
        policy_module=policy_module,
        model=model_enum,
        backend=backend,
        use_cache=use_cache,
        batch_size=batch_size,
    )

    # Convert to result rows
    results = [
        _create_result_row(row, classification_result, policy, model)
        for row, classification_result in zip(df.itertuples(), classification_results)
    ]
    
    results_df = pd.DataFrame(results)

    # Calculate and add sample weights
    labels = results_df['toxicity'].values
    weights = calculate_sample_weights(labels, true_positive_rate, sampling_strategy)
    results_df['sample_weight'] = weights
    results_df['sampling_strategy'] = sampling_strategy.value
    results_df['true_positive_rate'] = true_positive_rate

    # Add backend column if not API (for backwards compat)
    if backend != InferenceBackend.API:
        results_df['backend'] = backend.value
    
    print_summary_stats(results_df)
    
    # Generate output filename
    if output is None:
        workspace_dir = Path(__file__).parent
        output = get_cache_file_path(version, model, policy, split, sample_size, workspace_dir, sampling_strategy)
        results_dir = output.parent
        results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output}")
    results_df.to_parquet(output, index=False)
    print("Done!")
    
    print(f"\nResults DataFrame shape: {results_df.shape}")
    print(f"Columns: {list(results_df.columns)}")
    return results_df


@app.command(name="run-toxic-chat-evaluation")
def run_toxic_chat_evaluation_cli(
    sample_size: int = typer.Option(20, help='Number of examples to process'),
    version: Literal['toxicchat0124', 'toxicchat1123'] = typer.Option('toxicchat0124', help='Dataset version to use'),
    split: Literal['train', 'test'] = typer.Option('test', help='Dataset split to use'),
    policy: str = typer.Option('toxic_chat_claude_1', help='Policy name (e.g., toxic_chat_claude_1, spam)'),
    model: Literal['GPT_OSS_20B', 'GPT_OSS_safeguarded_20B'] = typer.Option('GPT_OSS_20B', help='Model to use'),
    output: Optional[Path] = typer.Option(None, help='Output parquet file path (default: auto-generated in results/)'),
    use_cache: bool = typer.Option(True, help='Load from cache if file exists'),
    sampling_strategy: SamplingStrategy = typer.Option(SamplingStrategy.NATURAL, help='Sampling strategy (natural/balanced/neyman)'),
    backend: InferenceBackend = typer.Option(InferenceBackend.API, help='Inference backend (api/local)'),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help='Batch size for LOCAL backend'),
):
    """CLI wrapper for run_toxic_chat_evaluation."""
    return run_toxic_chat_evaluation(
        sample_size=sample_size,
        version=version,
        split=split,
        policy=policy,
        model=model,
        output=output,
        use_cache=use_cache,
        sampling_strategy=sampling_strategy,
        backend=backend,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
