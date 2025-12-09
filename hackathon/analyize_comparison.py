"""Analyze comparison between two model classifications."""
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import pingouin as pg


def compare_results(results: list[dict], model1_key: str = None, model2_key: str = None) -> dict:
    """
    Analyze and compare classification results from two models.
    
    Args:
        results: List of dicts, each containing:
            - "text": the input text
            - model keys with float_label values
        model1_key: Key for first model (auto-detected if None)
        model2_key: Key for second model (auto-detected if None)
    
    Returns:
        Dictionary with analysis results
    """
    # Auto-detect model keys if not provided
    if model1_key is None or model2_key is None:
        if results:
            model_keys = [k for k in results[0].keys() if k.startswith("model_")]
            if len(model_keys) >= 2:
                model1_key = model_keys[0]
                model2_key = model_keys[1]
            else:
                raise ValueError(f"Could not find two model keys. Found: {model_keys}")
        else:
            raise ValueError("No results to analyze")
    
    print(f"Comparing: {model1_key} vs {model2_key}")
    
    # Extract scores (filter out None values)
    model1_scores = []
    model2_scores = []
    texts = []
    
    for r in results:
        s1 = r.get(model1_key)
        s2 = r.get(model2_key)
        if s1 is not None and s2 is not None:
            model1_scores.append(s1)
            model2_scores.append(s2)
            texts.append(r["text"])
    
    model1_scores = np.array(model1_scores)
    model2_scores = np.array(model2_scores)
    
    analysis = {}
    
    # =========================================================================
    # 1. Distribution of labels from each model
    # =========================================================================
    # Convert back to 1-5 scale for interpretability
    model1_likert = (model1_scores * 4 + 1).round().astype(int)
    model2_likert = (model2_scores * 4 + 1).round().astype(int)
    
    analysis["distribution"] = {
        "model1": {
            "name": model1_key,
            "counts": {int(k): int(v) for k, v in Counter(model1_likert).items()},
            "mean": float(np.mean(model1_scores)),
            "std": float(np.std(model1_scores)),
            "mean_likert": float(np.mean(model1_likert)),
        },
        "model2": {
            "name": model2_key, 
            "counts": {int(k): int(v) for k, v in Counter(model2_likert).items()},
            "mean": float(np.mean(model2_scores)),
            "std": float(np.std(model2_scores)),
            "mean_likert": float(np.mean(model2_likert)),
        }
    }
    
    # =========================================================================
    # 2. Typical differences
    # =========================================================================
    differences = model1_scores - model2_scores
    abs_differences = np.abs(differences)
    
    # Any difference at all (accounting for floating point)
    has_difference = abs_differences > 0.001
    
    analysis["differences"] = {
        "mean_difference": float(np.mean(differences)),  # Signed: positive = model1 higher
        "mean_abs_difference": float(np.mean(abs_differences)),
        "std_difference": float(np.std(differences)),
        "fraction_different": float(np.mean(has_difference)),
        "n_different": int(np.sum(has_difference)),
        "n_total": len(differences),
    }
    
    # =========================================================================
    # 3. Statistical tests
    # =========================================================================
    # Paired t-test (parametric) - tests if mean difference is significantly != 0
    if len(model1_scores) >= 2:
        t_stat, t_pvalue = stats.ttest_rel(model1_scores, model2_scores)
    else:
        t_stat, t_pvalue = np.nan, np.nan
    
    # Wilcoxon signed-rank test (non-parametric) - better for small samples
    # Tests if the distribution of differences is symmetric around zero
    if len(model1_scores) >= 5 and np.any(differences != 0):
        try:
            w_stat, w_pvalue = stats.wilcoxon(model1_scores, model2_scores)
        except ValueError:
            # All differences are zero
            w_stat, w_pvalue = np.nan, 1.0
    else:
        w_stat, w_pvalue = np.nan, np.nan
    
    # Sign test - simplest non-parametric, just counts which is higher
    n_model1_higher = np.sum(differences > 0.001)
    n_model2_higher = np.sum(differences < -0.001)
    n_ties = np.sum(np.abs(differences) <= 0.001)
    
    # Binomial test for sign test
    if n_model1_higher + n_model2_higher > 0:
        sign_pvalue = stats.binom_test(
            n_model1_higher, 
            n_model1_higher + n_model2_higher, 
            0.5, 
            alternative='two-sided'
        ) if hasattr(stats, 'binom_test') else stats.binomtest(
            n_model1_higher,
            n_model1_higher + n_model2_higher,
            0.5,
            alternative='two-sided'
        ).pvalue
    else:
        sign_pvalue = 1.0
    
    analysis["statistical_tests"] = {
        "paired_ttest": {
            "description": "Tests if mean difference is significantly different from 0",
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
            "p_value": float(t_pvalue) if not np.isnan(t_pvalue) else None,
            "interpretation": "Assumes normal distribution of differences",
        },
        "wilcoxon_signed_rank": {
            "description": "Non-parametric test for paired samples",
            "w_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "p_value": float(w_pvalue) if not np.isnan(w_pvalue) else None,
            "interpretation": "Better for small samples, doesn't assume normality",
        },
        "sign_test": {
            "description": "Counts which model scores higher more often",
            "n_model1_higher": int(n_model1_higher),
            "n_model2_higher": int(n_model2_higher),
            "n_ties": int(n_ties),
            "p_value": float(sign_pvalue),
            "interpretation": "Simplest test - just compares direction of differences",
        }
    }
    
    # =========================================================================
    # 5. Inter-rater reliability (ICC and correlations)
    # =========================================================================
    # Prepare data for ICC calculation (needs long format)
    n = len(model1_scores)
    icc_data = pd.DataFrame({
        'targets': list(range(n)) * 2,
        'raters': ['model1'] * n + ['model2'] * n,
        'ratings': np.concatenate([model1_scores, model2_scores])
    })
    
    # Calculate ICC using pingouin
    icc_results = pg.intraclass_corr(data=icc_data, targets='targets', 
                                      raters='raters', ratings='ratings')
    
    # Extract ICC3 (two-way mixed, absolute agreement, single rater)
    icc3_row = icc_results[icc_results['Type'] == 'ICC3'].iloc[0]
    
    # Calculate correlations
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(model1_scores, model2_scores)
    r_spearman, p_spearman = spearmanr(model1_scores, model2_scores)
    
    analysis["inter_rater_reliability"] = {
        "icc3": {
            "description": "ICC(3,1): Two-way mixed effects, absolute agreement, single rater",
            "icc": float(icc3_row['ICC']),
            "ci_95_lower": float(icc3_row['CI95%'][0]),
            "ci_95_upper": float(icc3_row['CI95%'][1]),
            "p_value": float(icc3_row['pval']),
            "interpretation": _interpret_icc(float(icc3_row['ICC'])),
            "meaning": "Proportion of variance due to true differences between items (vs rater disagreement)"
        },
        "pearson_correlation": {
            "description": "Linear correlation between ratings",
            "r": float(r_pearson),
            "p_value": float(p_pearson),
            "interpretation": "Measures linear relationship (doesn't penalize systematic bias)"
        },
        "spearman_correlation": {
            "description": "Rank-order correlation between ratings",
            "rho": float(r_spearman),
            "p_value": float(p_spearman),
            "interpretation": "Measures agreement on ranking (robust to outliers)"
        }
    }
    
    # =========================================================================
    # 4. Instances with largest differences
    # =========================================================================
    # Sort by absolute difference
    sorted_indices = np.argsort(-abs_differences)
    
    most_different = []
    for idx in sorted_indices[:10]:  # Top 10 most different
        most_different.append({
            "text": texts[idx],
            "model1_score": float(model1_scores[idx]),
            "model2_score": float(model2_scores[idx]),
            "model1_likert": int(model1_likert[idx]),
            "model2_likert": int(model2_likert[idx]),
            "difference": float(differences[idx]),
            "abs_difference": float(abs_differences[idx]),
        })
    
    analysis["most_different_instances"] = most_different
    
    return analysis


def _short_model_name(name: str) -> str:
    """Extract a short model name from the full key."""
    # e.g., "model_openai/gpt-oss-20b" -> "gpt-oss-20b"
    if name.startswith("model_"):
        name = name[6:]
    if "/" in name:
        name = name.split("/")[-1]
    return name


def _interpret_icc(icc: float) -> str:
    """Interpret ICC value according to standard guidelines."""
    if icc < 0.5:
        return "Poor reliability"
    elif icc < 0.75:
        return "Moderate reliability"
    elif icc < 0.9:
        return "Good reliability"
    else:
        return "Excellent reliability"


def print_analysis(analysis: dict):
    """Pretty print the analysis results."""
    model1_name = _short_model_name(analysis["distribution"]["model1"]["name"])
    model2_name = _short_model_name(analysis["distribution"]["model2"]["name"])
    
    print("=" * 70)
    print("DISTRIBUTION OF LABELS")
    print("=" * 70)
    
    for model_key in ["model1", "model2"]:
        m = analysis["distribution"][model_key]
        print(f"\n{m['name']}:")
        print(f"  Likert counts: {m['counts']}")
        print(f"  Mean (0-1 scale): {m['mean']:.3f}")
        print(f"  Mean (1-5 scale): {m['mean_likert']:.2f}")
        print(f"  Std (0-1 scale): {m['std']:.3f}")
    
    print("\n" + "=" * 70)
    print("DIFFERENCES BETWEEN MODELS")
    print("=" * 70)
    
    d = analysis["differences"]
    print(f"\nMean difference ({model1_name} - {model2_name}): {d['mean_difference']:+.3f}")
    print(f"Mean absolute difference: {d['mean_abs_difference']:.3f}")
    print(f"Std of differences: {d['std_difference']:.3f}")
    print(f"Fraction with any difference: {d['fraction_different']:.1%} ({d['n_different']}/{d['n_total']})")
    
    print("\n" + "=" * 70)
    print("INTER-RATER RELIABILITY")
    print("=" * 70)
    
    irr = analysis["inter_rater_reliability"]
    
    print("\nICC(3,1) - Intraclass Correlation Coefficient:")
    icc_info = irr["icc3"]
    print(f"  ICC = {icc_info['icc']:.3f} (95% CI: [{icc_info['ci_95_lower']:.3f}, {icc_info['ci_95_upper']:.3f}])")
    print(f"  Interpretation: {icc_info['interpretation']}")
    print(f"  p-value: {icc_info['p_value']:.4f}")
    print(f"  → {icc_info['icc']*100:.1f}% of variance is true differences between items")
    print(f"  → {(1-icc_info['icc'])*100:.1f}% of variance is rater disagreement")
    
    print("\nCorrelations:")
    print(f"  Pearson r = {irr['pearson_correlation']['r']:.3f} (p = {irr['pearson_correlation']['p_value']:.4f})")
    print(f"  Spearman ρ = {irr['spearman_correlation']['rho']:.3f} (p = {irr['spearman_correlation']['p_value']:.4f})")
    
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Mean Differences)")
    print("=" * 70)
    
    for test_name, test in analysis["statistical_tests"].items():
        print(f"\n{test_name}:")
        print(f"  {test['description']}")
        if test.get('p_value') is not None:
            p = test['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  p-value: {p:.4f} {sig}")
        if 'n_model1_higher' in test:
            print(f"  {model1_name} higher: {test['n_model1_higher']}, {model2_name} higher: {test['n_model2_higher']}, Ties: {test['n_ties']}")
    
    print("\n" + "=" * 70)
    print("MOST DIFFERENT INSTANCES")
    print("=" * 70)
    
    for i, inst in enumerate(analysis["most_different_instances"], 1):
        if inst["abs_difference"] < 0.001:
            break
        max_len = 120
        print(f"\n{i}. \"{inst['text'][:max_len]}...\"" if len(inst['text']) > max_len else f"\n{i}. \"{inst['text']}\"")
        print(f"   {model1_name}: {inst['model1_likert']}/5 ({inst['model1_score']:.2f})")
        print(f"   {model2_name}: {inst['model2_likert']}/5 ({inst['model2_score']:.2f})")
        print(f"   Difference: {inst['difference']:+.2f}")


if __name__ == "__main__":
    # Example usage with mock data for testing
    mock_results = [
        {"text": "Test 1", "model_gpt-oss-20b": 0.5, "model_gpt-oss-safeguarded-20b": 0.75},
        {"text": "Test 2", "model_gpt-oss-20b": 0.25, "model_gpt-oss-safeguarded-20b": 0.25},
        {"text": "Test 3", "model_gpt-oss-20b": 1.0, "model_gpt-oss-safeguarded-20b": 0.5},
    ]
    
    analysis = compare_results(mock_results)
    print_analysis(analysis)
