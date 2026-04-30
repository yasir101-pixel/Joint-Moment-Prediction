import numpy as np
from scipy.stats import pearsonr

# ─────────────────────────────────────────
# METRICS (following Liew 2021)
# ─────────────────────────────────────────

def rmse(y_true, y_pred):
    """Root Mean Squared Error (Nm/kg)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def rel_rmse(y_true, y_pred):
    """
    Relative RMSE (%) normalized by average peak-to-peak amplitude.
    Following Liew 2021 Eq. 2.
    """
    rms = rmse(y_true, y_pred)
    peak_to_peak_obs = y_true.max(axis=0) - y_true.min(axis=0)
    peak_to_peak_pred = y_pred.max(axis=0) - y_pred.min(axis=0)
    avg_ptp = 0.5 * (peak_to_peak_obs + peak_to_peak_pred)
    avg_ptp = np.where(avg_ptp == 0, 1, avg_ptp)
    return (rms / avg_ptp) * 100.0


def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient per output."""
    n_outputs = y_true.shape[1]
    correlations = []
    for i in range(n_outputs):
        r, _ = pearsonr(y_true[:, i], y_pred[:, i])
        correlations.append(r)
    return np.array(correlations)


def evaluate_model(y_true, y_pred, moment_names=None):
    """
    Compute RMSE, relRMSE, and Pearson r for all joint moments.
    Returns a summary dictionary.
    """
    rms = rmse(y_true, y_pred)
    rel_rms = rel_rmse(y_true, y_pred)
    corr = pearson_r(y_true, y_pred)

    n_outputs = y_true.shape[1]
    if moment_names is None:
        moment_names = [f'moment_{i}' for i in range(n_outputs)]

    results = {}
    for i, name in enumerate(moment_names):
        results[name] = {
            'RMSE': round(float(rms[i]), 4),
            'relRMSE': round(float(rel_rms[i]), 2),
            'pearson_r': round(float(corr[i]), 4)
        }

    # Summary averages
    results['AVERAGE'] = {
        'RMSE': round(float(rms.mean()), 4),
        'relRMSE': round(float(rel_rms.mean()), 2),
        'pearson_r': round(float(corr.mean()), 4)
    }

    return results


def print_results(results, model_name='Model'):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Results: {model_name}")
    print(f"{'='*60}")
    print(f"{'Moment':<35} {'RMSE':>8} {'relRMSE':>10} {'r':>8}")
    print(f"{'-'*60}")
    for name, metrics in results.items():
        print(f"{name:<35} {metrics['RMSE']:>8.4f} "
              f"{metrics['relRMSE']:>9.2f}% {metrics['pearson_r']:>8.4f}")
    print(f"{'='*60}")


def save_results(results, model_name, output_path):
    """Save results to CSV."""
    import pandas as pd
    rows = []
    for moment, metrics in results.items():
        rows.append({
            'model': model_name,
            'moment': moment,
            'RMSE': metrics['RMSE'],
            'relRMSE': metrics['relRMSE'],
            'pearson_r': metrics['pearson_r']
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df


if __name__ == '__main__':
    # Quick test
    y_true = np.random.randn(100, 10)
    y_pred = y_true + np.random.randn(100, 10) * 0.1
    results = evaluate_model(y_true, y_pred)
    print_results(results, model_name='Test')
    print("evaluate.py OK!")