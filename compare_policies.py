import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from workspace.run_toxic_chat import run_toxic_chat_evaluation
from workspace.model_predict import Model
from workspace.common.texrendering import generate_latex_table


def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrapped confidence interval for a metric."""
    n_samples = len(y_true)
    bootstrap_scores = []
    
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, ci_lower, ci_upper


def calculate_metrics_with_ci(df, n_bootstrap=1000):
    """Calculate metrics with bootstrapped confidence intervals."""
    y_true = df['toxicity'].astype(int).values
    y_pred = df['predicted_toxic'].astype(int).values
    parsed = df['parsed_successfully'].values
    
    # Calculate metrics with bootstrap CI
    precision_mean, precision_lower, precision_upper = bootstrap_metric(
        y_true, y_pred, 
        lambda yt, yp: precision_score(yt, yp, zero_division=0),
        n_bootstrap=n_bootstrap
    )
    
    recall_mean, recall_lower, recall_upper = bootstrap_metric(
        y_true, y_pred,
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        n_bootstrap=n_bootstrap
    )
    
    f1_mean, f1_lower, f1_upper = bootstrap_metric(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n_bootstrap=n_bootstrap
    )
    
    # Parse rate bootstrap (proportion of successfully parsed)
    parse_mean, parse_lower, parse_upper = bootstrap_metric(
        np.arange(len(parsed)), parsed,
        lambda idx, p: np.mean(p),
        n_bootstrap=n_bootstrap
    )
    
    return {
        'precision': (precision_mean, precision_lower, precision_upper),
        'recall': (recall_mean, recall_lower, recall_upper),
        'f1': (f1_mean, f1_lower, f1_upper),
        'parse_rate': (parse_mean, parse_lower, parse_upper),
    }


def format_metric_with_ci(mean, lower, upper, as_percent=False):
    """Format metric with confidence interval."""
    if as_percent:
        mean_str = f"{mean*100:.1f}"
        ci_range = max(mean - lower, upper - mean) * 100
        return f"{mean_str} $\\pm$ {ci_range:.1f}"
    else:
        mean_str = f"{mean:.3f}"
        ci_range = max(mean - lower, upper - mean)
        return f"{mean_str} $\\pm$ {ci_range:.3f}"


def run_model_evaluations(
    sample_size: int,
    use_cache: bool = True,
    version: str = 'toxicchat0124',
    split: str = 'test',
    policy: str = 'toxic_chat_claude_1',
):
    """Run toxic chat evaluation for both base and safeguarded models.
    
    Returns:
        tuple: (base_df, safeguard_df) DataFrames with evaluation results
    """
    base_df = run_toxic_chat_evaluation(
        sample_size=sample_size,
        version=version,
        split=split,
        policy=policy,
        model=Model.GPT_OSS_20B.name,
        use_cache=use_cache,
    )
    
    safeguard_df = run_toxic_chat_evaluation(
        sample_size=sample_size,
        version=version,
        split=split,
        policy=policy,
        model=Model.GPT_OSS_safeguarded_20B.name,
        use_cache=use_cache,
    )
    
    return base_df, safeguard_df


def run_policy_evaluations(
    sample_size: int,
    use_cache: bool = True,
    version: str = 'toxicchat0124',
    split: str = 'test',
    policies: list = None,
):
    """Run toxic chat evaluation for multiple policies on both models.
    
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
        )
        results[policy] = (base_df, safeguard_df)
    
    return results


def run_all_policies(
    sample_size: int = 300,
    use_cache: bool = True,
):
    """Run evaluations for both models and concatenate results."""
    base_df, safeguard_df = run_model_evaluations(
        sample_size=sample_size,
        use_cache=use_cache,
    )
    return pd.concat([base_df, safeguard_df])


def generate_comparison_table(sample_size: int = 1000, use_cache: bool = True, n_bootstrap: int = 1000):
    """Generate table data comparing policies across models.
    
    Returns:
        tuple: (rows, latex_table) where rows is the data structure and latex_table is the formatted LaTeX
    """
    
    # Run evaluations for all policies
    policy_results = run_policy_evaluations(
        sample_size=sample_size,
        use_cache=use_cache,
        policies=[
            'toxic_chat_claude_1', 
            #'toxic_simple'
        ]
    )
    
    # Policy display names
    policy_names = {
        'toxic_chat_claude_1': 'ToxicChat Claude 1',
        'toxic_simple': 'Toxic Simple'
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
                'Base': {
                    'F1': format_metric_with_ci(*base_metrics['f1']),
                    'Precision': format_metric_with_ci(*base_metrics['precision']),
                    'Recall': format_metric_with_ci(*base_metrics['recall']),
                    'Parse Rate': format_metric_with_ci(*base_metrics['parse_rate'], as_percent=True),
                },
                'Safeguard': {
                    'F1': format_metric_with_ci(*safeguard_metrics['f1']),
                    'Precision': format_metric_with_ci(*safeguard_metrics['precision']),
                    'Recall': format_metric_with_ci(*safeguard_metrics['recall']),
                    'Parse Rate': format_metric_with_ci(*safeguard_metrics['parse_rate'], as_percent=True),
                }
            }
        }
        rows.append(row)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(
        rows=rows,
        full_width=False,
        caption_content="",
        label="tab:policy_comparison",
        caption_at_top=False,
        resize_to_fit=False
    )
    
    return rows, latex_table


def main():
    rows, latex_table = generate_comparison_table(
        sample_size=2000, 
        use_cache=True, 
        n_bootstrap=1000
    )
    print("\n" + "="*80)
    print("GENERATED LATEX TABLE")
    print("="*80)
    print(latex_table)
    print("\n")
    
    # Save LaTeX table to file
    from pathlib import Path
    output_file = "workspace/tables/policy_comparison.tex"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table)
    print(f"LaTeX table saved to: {output_file}")
    
    # Also generate HTML version using the same data
    from workspace.common.htmlrendering import generate_html_table
    html_table = generate_html_table(rows=rows)
    
    # Save HTML table
    html_output_file = "workspace/tables/policy_comparison.html"
    html_output_path = Path(html_output_file)
    with open(html_output_path, "w") as f:
        f.write(html_table)
    print(f"HTML table saved to: {html_output_file}")


if __name__ == "__main__":
    main()