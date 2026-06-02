"""
Statistical significance tests for model comparison.

Implements:
- McNemar's test (paired): tests if two models make different mistakes
- Bootstrap confidence intervals for accuracy
- Wilcoxon signed-rank test across cross-validation folds
"""
import numpy as np
from typing import Tuple, List, Optional
from scipy import stats


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float]:
    """McNemar's test for paired nominal data.

    Compares whether two classifiers make different errors on the same test set.
    H0: the two classifiers have the same error rate.

    Returns:
        (statistic, p_value)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # n01: A wrong, B correct
    n01 = np.sum(~correct_a & correct_b)
    # n10: A correct, B wrong
    n10 = np.sum(correct_a & ~correct_b)

    # Continuity correction (Edwards correction)
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    p_value = 1.0 - stats.chi2.cdf(stat, df=1)

    return stat, p_value


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for accuracy.

    Returns:
        (lower_bound, accuracy_mean, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    bootstrapped_accs = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        acc = np.mean(y_pred[idx] == y_true[idx])
        bootstrapped_accs.append(acc)

    lower = np.percentile(bootstrapped_accs, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_accs, 100 * (1 - alpha / 2))
    mean_acc = np.mean(bootstrapped_accs)

    return lower, mean_acc, upper


def compare_models_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap test for the difference in accuracy between two models.

    Returns:
        (p_value, mean_difference)
        where mean_difference = acc_a - acc_b
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        acc_a = np.mean(y_pred_a[idx] == y_true[idx])
        acc_b = np.mean(y_pred_b[idx] == y_true[idx])
        diffs.append(acc_a - acc_b)

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)

    # Two-sided p-value: proportion of bootstrap samples where
    # difference crosses zero (or is more extreme)
    p_value = np.mean(np.abs(diffs - mean_diff) >= np.abs(mean_diff))

    return p_value, mean_diff


def wilcoxon_signed_rank_test(
    fold_metrics_a: List[float],
    fold_metrics_b: List[float],
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired cross-validation metrics.

    Use when you have accuracy from the same fold for two models.
    
    H0: the median difference between paired observations is zero.

    Returns:
        (statistic, p_value)
    """
    return stats.wilcoxon(fold_metrics_a, fold_metrics_b)


def format_significance(p_value: float, alpha: float = 0.05) -> str:
    """Format a p-value with significance stars."""
    if p_value < 0.001:
        return f"p < 0.001 ***"
    elif p_value < 0.01:
        return f"p = {p_value:.4f} **"
    elif p_value < 0.05:
        return f"p = {p_value:.4f} *"
    else:
        return f"p = {p_value:.4f} (n.s.)"


def compute_all_pairwise_tests(
    model_predictions: dict,  # {model_name: (y_true, y_pred)}
    alpha: float = 0.05,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute McNemar and bootstrap tests for all model pairs.

    Returns a dict mapping (model_a, model_b) -> dict of test results.
    """
    results = {}
    model_names = list(model_predictions.keys())

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            y_true_a, y_pred_a = model_predictions[name_a]
            y_true_b, y_pred_b = model_predictions[name_b]

            # Ensure same test set
            assert len(y_true_a) == len(y_true_b)
            assert np.array_equal(y_true_a, y_true_b), "Different test sets!"

            # McNemar
            mcn_stat, mcn_p = mcnemar_test(y_true_a, y_pred_a, y_pred_b)

            # Bootstrap
            boot_p, mean_diff = compare_models_bootstrap(
                y_true_a, y_pred_a, y_pred_b, n_bootstrap=n_bootstrap, seed=seed
            )

            # Bootstrap CI for each
            ci_a_low, ci_a_mean, ci_a_high = bootstrap_accuracy_ci(
                y_true_a, y_pred_a, n_bootstrap=n_bootstrap, seed=seed
            )
            ci_b_low, ci_b_mean, ci_b_high = bootstrap_accuracy_ci(
                y_true_b, y_pred_b, n_bootstrap=n_bootstrap, seed=seed
            )

            results[(name_a, name_b)] = {
                "acc_a": ci_a_mean,
                "acc_b": ci_b_mean,
                "ci_a": (ci_a_low, ci_a_high),
                "ci_b": (ci_b_low, ci_b_high),
                "mean_diff": mean_diff,
                "mcnemar_stat": mcn_stat,
                "mcnemar_p": mcn_p,
                "bootstrap_p": boot_p,
                "significant_mcnemar": mcn_p < alpha,
                "significant_bootstrap": boot_p < alpha,
            }

    return results
