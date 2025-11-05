"""
Statistical metrics utilities with bootstrap confidence intervals.

This module provides reusable functions for calculating classification metrics
(precision, recall, F1) with bootstrapped confidence intervals.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, confidence_level=0.95, random_seed=42, sample_weights=None):
    """
    Calculate bootstrapped confidence interval for any statistic.
    
    Args:
        data: Array-like or tuple of arrays to resample
        statistic_fn: Function that takes resampled data and returns a scalar statistic
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        sample_weights: Optional array of sample weights for weighted bootstrap resampling
        
    Returns:
        tuple: (mean, lower_ci, upper_ci)
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> mean, lower, upper = bootstrap_ci(
        ...     (y_true, y_pred),
        ...     lambda data: f1_score(data[0], data[1], zero_division=0)
        ... )
    """
    # Handle single array or tuple of arrays
    if not isinstance(data, tuple):
        data = (data,)
    
    n_samples = len(data[0])
    bootstrap_scores = []
    
    rng = np.random.RandomState(random_seed)
    
    # Calculate sampling probabilities for weighted bootstrap
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)
        # Normalize weights to probabilities
        sampling_probs = sample_weights / sample_weights.sum()
    else:
        sampling_probs = None
    
    for _ in range(n_bootstrap):
        # Resample with replacement (weighted if weights provided)
        indices = rng.choice(n_samples, size=n_samples, replace=True, p=sampling_probs)
        resampled_data = tuple(arr[indices] for arr in data)
        
        # If weights provided, also resample the weights
        if sample_weights is not None:
            resampled_weights = sample_weights[indices]
            resampled_data = resampled_data + (resampled_weights,)
        
        # Calculate statistic
        score = statistic_fn(resampled_data)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, ci_lower, ci_upper


def calculate_classification_metrics_with_ci(y_true, y_pred, n_bootstrap=1000, random_seed=42, sample_weights=None):
    """
    Calculate precision, recall, and F1 with bootstrapped confidence intervals.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        sample_weights: Optional array of sample weights for weighted metrics
        
    Returns:
        dict: Dictionary with keys 'precision', 'recall', 'f1', each containing
              tuple of (mean, lower_ci, upper_ci)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Define metrics to calculate
    if sample_weights is not None:
        # Weighted metrics: data tuple includes weights as third element
        metrics = {
            'precision': lambda data: precision_score(data[0], data[1], sample_weight=data[2], zero_division=0),
            'recall': lambda data: recall_score(data[0], data[1], sample_weight=data[2], zero_division=0),
            'f1': lambda data: f1_score(data[0], data[1], sample_weight=data[2], zero_division=0),
        }
    else:
        # Unweighted metrics
        metrics = {
            'precision': lambda data: precision_score(data[0], data[1], zero_division=0),
            'recall': lambda data: recall_score(data[0], data[1], zero_division=0),
            'f1': lambda data: f1_score(data[0], data[1], zero_division=0),
        }
    
    # Calculate all metrics with bootstrap CI
    results = {}
    for metric_name, metric_fn in metrics.items():
        results[metric_name] = bootstrap_ci(
            (y_true, y_pred),
            metric_fn,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
            sample_weights=sample_weights
        )
    
    return results


def calculate_proportion_with_ci(values, n_bootstrap=1000, random_seed=42):
    """
    Calculate proportion (mean) with bootstrapped confidence interval.
    
    Useful for metrics like parse rate, success rate, etc.
    
    Args:
        values: Binary or boolean array
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (mean, lower_ci, upper_ci)
    """
    values = np.asarray(values)
    return bootstrap_ci(
        values,
        lambda data: np.mean(data[0]),
        n_bootstrap=n_bootstrap,
        random_seed=random_seed
    )


def format_metric_with_ci(mean, lower, upper, as_percent=False, precision=None):
    """
    Format metric with confidence interval as ± range.
    
    Args:
        mean: Mean value
        lower: Lower confidence bound
        upper: Upper confidence bound
        as_percent: If True, format as percentage with 1 decimal place
        precision: Number of decimal places (default: 1 for percent, 3 for decimal)
        
    Returns:
        str: Formatted string like "0.850 ± 0.023" or "85.0 ± 2.3"
    """
    if as_percent:
        if precision is None:
            precision = 1
        mean_val = mean * 100
        ci_range = max(mean - lower, upper - mean) * 100
        return f"{mean_val:.{precision}f} $\\pm$ {ci_range:.{precision}f}"
    else:
        if precision is None:
            precision = 3
        ci_range = max(mean - lower, upper - mean)
        return f"{mean:.{precision}f} $\\pm$ {ci_range:.{precision}f}"

