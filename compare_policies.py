import pandas as pd
from pathlib import Path
from workspace.run_toxic_chat import run_toxic_chat_evaluation
from workspace.model_predict import Model, InferenceBackend
from workspace.common.htmlrendering import generate_html_table
from workspace.metrics import (
    calculate_classification_metrics_with_ci,
    calculate_proportion_with_ci,
    format_metric_with_ci,
)


def calculate_metrics_with_ci(df, n_bootstrap=1000):
    """
    Calculate classification and parse rate metrics with bootstrapped CIs.
    
    Args:
        df: DataFrame with columns 'toxicity', 'predicted_toxic', 'parsed_successfully'
            Optional: 'sample_weight' for weighted metrics
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        dict: Dictionary with keys 'precision', 'recall', 'f1', 'parse_rate',
              each containing tuple of (mean, lower_ci, upper_ci)
    """
    y_true = df['toxicity'].astype(int).values
    y_pred = df['predicted_toxic'].astype(int).values
    parsed = df['parsed_successfully'].values
    
    # Extract sample weights if present
    sample_weights = None
    if 'sample_weight' in df.columns:
        sample_weights = df['sample_weight'].values
    
    # Calculate classification metrics (with weights if available)
    metrics = calculate_classification_metrics_with_ci(
        y_true, y_pred, n_bootstrap=n_bootstrap, sample_weights=sample_weights
    )
    
    # Add parse rate (unweighted - parsing success is not affected by class imbalance)
    metrics['parse_rate'] = calculate_proportion_with_ci(
        parsed, n_bootstrap=n_bootstrap
    )
    
    return metrics


def run_model_evaluations(
    sample_size: int,
    use_cache: bool = True,
    version: str = 'toxicchat0124',
    split: str = 'test',
    policy: str = 'toxic_chat_claude_1',
    sampling_strategy: str = 'natural',
    backend: InferenceBackend = InferenceBackend.LOCAL,
    batch_size: int = 8,
):
    """Run toxic chat evaluation for both base and safeguarded models.

    Args:
        sample_size: Number of examples to sample
        use_cache: Whether to use cached results
        version: Dataset version
        split: Dataset split
        policy: Policy name
        sampling_strategy: Sampling strategy (natural/balanced/neyman)
        backend: Inference backend (API or LOCAL)
        batch_size: Batch size for LOCAL backend

    Returns:
        tuple: (base_df, safeguard_df) DataFrames with evaluation results
    """
    from workspace.run_toxic_chat import SamplingStrategy

    # Convert string to enum
    strategy_enum = SamplingStrategy(sampling_strategy)

    base_df = run_toxic_chat_evaluation(
        sample_size=sample_size,
        version=version,
        split=split,
        policy=policy,
        model=Model.GPT_OSS_20B.name,
        use_cache=use_cache,
        sampling_strategy=strategy_enum,
        backend=backend,
        batch_size=batch_size,
    )

    safeguard_df = run_toxic_chat_evaluation(
        sample_size=sample_size,
        version=version,
        split=split,
        policy=policy,
        model=Model.GPT_OSS_safeguarded_20B.name,
        use_cache=use_cache,
        sampling_strategy=strategy_enum,
        backend=backend,
        batch_size=batch_size,
    )

    return base_df, safeguard_df


def run_policy_evaluations(
    sample_size: int,
    use_cache: bool = True,
    version: str = 'toxicchat0124',
    split: str = 'test',
    policies: list = None,
    sampling_strategy: str = 'natural',
    backend: InferenceBackend = InferenceBackend.LOCAL,
    batch_size: int = 8,
):
    """Run toxic chat evaluation for multiple policies on both models.

    Args:
        sample_size: Number of examples to sample
        use_cache: Whether to use cached results
        version: Dataset version
        split: Dataset split
        policies: List of policy names
        sampling_strategy: Sampling strategy (natural/balanced/neyman)
        backend: Inference backend (API or LOCAL)
        batch_size: Batch size for LOCAL backend

    Returns:
        dict: {policy_name: (base_df, safeguard_df)} for each policy
    """
    if policies is None:
        policies = ['toxic_chat_claude_1', 'toxic_simple']

    results = {}
    for policy in policies:
        base_df, safeguard_df = run_model_evaluations(
            sample_size=sample_size,
            use_cache=use_cache,
            version=version,
            split=split,
            policy=policy,
            sampling_strategy=sampling_strategy,
            backend=backend,
            batch_size=batch_size,
        )
        results[policy] = (base_df, safeguard_df)

    return results


def run_all_policies(
    sample_size: int = 300,
    use_cache: bool = True,
    sampling_strategy: str = 'natural',
    backend: InferenceBackend = InferenceBackend.LOCAL,
    batch_size: int = 8,
):
    """Run evaluations for both models and concatenate results."""
    base_df, safeguard_df = run_model_evaluations(
        sample_size=sample_size,
        use_cache=use_cache,
        sampling_strategy=sampling_strategy,
        backend=backend,
        batch_size=batch_size,
    )
    return pd.concat([base_df, safeguard_df])


def generate_comparison_table(
    sample_size: int = 1000,
    use_cache: bool = True,
    n_bootstrap: int = 1000,
    sampling_strategy: str = 'natural',
    backend: InferenceBackend = InferenceBackend.LOCAL,
    batch_size: int = 8,
):
    """Generate HTML table comparing policies across models.

    Args:
        sample_size: Number of examples to sample
        use_cache: Whether to use cached results
        n_bootstrap: Number of bootstrap iterations
        sampling_strategy: Sampling strategy (natural/balanced/neyman)
        backend: Inference backend (API or LOCAL)
        batch_size: Batch size for LOCAL backend

    Returns:
        str: HTML table string
    """
    # Run evaluations for all policies
    policy_results = run_policy_evaluations(
        sample_size=sample_size,
        use_cache=use_cache,
        policies=[
            'toxic_chat_claude_1',
            'toxic_simple',
            'toxic_known_dataset',
        ],
        sampling_strategy=sampling_strategy,
        backend=backend,
        batch_size=batch_size,
    )
    
    # Policy display names
    policy_names = {
        'toxic_chat_claude_1': 'ToxicChat Claude 1',
        'toxic_simple': 'Toxic Simple',
        'toxic_known_dataset': 'ToxicChat Known Dataset',
    }
    
    def format_metrics_dict(metrics):
        """Helper to format metrics dictionary with consistent structure."""
        return {
            'F1': format_metric_with_ci(*metrics['f1']),
            'Precision': format_metric_with_ci(*metrics['precision']),
            'Recall': format_metric_with_ci(*metrics['recall']),
            'Parse Rate': format_metric_with_ci(*metrics['parse_rate'], as_percent=True),
        }
    
    # Create table rows with nested structure
    rows = []
    for policy_key, (base_df, safeguard_df) in policy_results.items():
        # Calculate metrics for each model
        base_metrics = calculate_metrics_with_ci(base_df, n_bootstrap=n_bootstrap)
        safeguard_metrics = calculate_metrics_with_ci(safeguard_df, n_bootstrap=n_bootstrap)
        
        row = {
            'Policy': policy_names.get(policy_key, policy_key),
            'GPT-OSS-20B': {
                'Base': format_metrics_dict(base_metrics),
                'Safeguard': format_metrics_dict(safeguard_metrics),
            }
        }
        rows.append(row)
    
    # Generate HTML table
    return generate_html_table(rows=rows)


def main():
    html_table = generate_comparison_table(
        sample_size=2000, 
        use_cache=True, 
        n_bootstrap=1000,
        sampling_strategy='natural'
    )
    
    # Save HTML table
    output_file = Path("workspace/tables/policy_comparison.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_table)
    
    print(f"✓ HTML table saved to: {output_file}")


if __name__ == "__main__":
    main()