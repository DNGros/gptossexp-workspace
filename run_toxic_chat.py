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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from .classify import classify
from .model_predict import Model

app = typer.Typer()


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
    workspace_dir: Optional[Path] = None
) -> Path:
    """Generate the expected cache file path for a given configuration."""
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent
    results_dir = workspace_dir / "results" / version / model
    base_filename = f"toxic_chat_{policy}_{split}_n{sample_size}.parquet"
    return results_dir / base_filename


def print_summary_stats(results_df: pd.DataFrame):
    """Print summary statistics and confusion matrix."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total examples processed: {len(results_df)}")
    
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


def run_toxic_chat_evaluation(
    sample_size: int = 20,
    version: Literal['toxicchat0124', 'toxicchat1123'] = 'toxicchat0124',
    split: Literal['train', 'test'] = 'test',
    policy: str = 'toxic_chat_claude_1',
    model: Literal['GPT_OSS_20B', 'GPT_OSS_safeguarded_20B'] = 'GPT_OSS_20B',
    output: Optional[Path] = None,
    use_cache: bool = True
):
    """Run toxicity classification on ToxicChat dataset."""
    # Check cache first if use_cache is True and output is not explicitly provided
    if use_cache and output is None:
        workspace_dir = Path(__file__).parent
        cache_file = get_cache_file_path(version, model, policy, split, sample_size, workspace_dir)
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
    
    dataset = load_dataset("lmsys/toxic-chat", version, split=split)
    
    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Sampling {sample_size} examples...")
    
    df = dataset.to_pandas()
    if len(df) > sample_size:
        # Use stable sampling: shuffle once, then take first n items
        # This ensures that increasing sample_size preserves previous samples
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df = df_shuffled.head(sample_size).reset_index(drop=True)
    
    print(f"Processing {len(df)} examples...")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
        classification_result = classify(
            text=row['user_input'],
            policy_module=policy_module,
            model=model_enum
        )
        
        result_row = {
            'conv_id': row['conv_id'],
            'user_input': row['user_input'],
            'model_output': row['model_output'],
            'human_annotation': row['human_annotation'],
            'toxicity': row['toxicity'],
            'jailbreaking': row['jailbreaking'],
            'openai_moderation': row['openai_moderation'],
            'policy_name': policy,
            'model_name': model,
            'predicted_toxic': classification_result.binary_label,
            'predicted_label': classification_result.fine_grain_label,
            'parsed_successfully': classification_result.parsed_successfully,
            'raw_model_response': classification_result.model_response.response,
            'model_reasoning': classification_result.model_response.reasoning,
            'model_used': classification_result.model_response.model,
            'matches_ground_truth': (
                classification_result.binary_label == (row['toxicity'] == 1)
            ),
        }
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    
    print_summary_stats(results_df)
    
    # Generate output filename
    if output is None:
        workspace_dir = Path(__file__).parent
        output = get_cache_file_path(version, model, policy, split, sample_size, workspace_dir)
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
    use_cache: bool = typer.Option(True, help='Load from cache if file exists')
):
    """CLI wrapper for run_toxic_chat_evaluation."""
    return run_toxic_chat_evaluation(
        sample_size=sample_size,
        version=version,
        split=split,
        policy=policy,
        model=model,
        output=output,
        use_cache=use_cache
    )


if __name__ == "__main__":
    app()
