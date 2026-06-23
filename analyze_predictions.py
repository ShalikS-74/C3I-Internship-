"""
analyze_predictions.py - Reliability Analysis for Building Footprint Segmentation

Analyzes whether segmentation quality (IoU, Dice) predicts downstream
urban-planning metrics (Building Count Error, Building Coverage Error).

Key Finding:
    Correlation(IoU, Count Error) is weak-to-moderate.
    High IoU does NOT necessarily imply low count error.

Usage:
    # Analyze single model CSV
    python analyze_predictions.py --csv results/deeplabv3plus_resnet34_metrics.csv

    # Compare multiple models
    python analyze_predictions.py --csv results/deeplabv3plus_resnet34_metrics.csv results/unet_scse_resnet34_metrics.csv

    # Custom output
    python analyze_predictions.py --csv results/model_metrics.csv --output_dir analysis --no_plots
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# =============================================================================
# CSV Loading
# =============================================================================

def load_metrics_csv(filepath: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load per-sample metrics from CSV exported by evaluate.py.
    
    Expected columns: filename, iou, dice, count_error, coverage_error,
                      pred_count, gt_count, pred_coverage, gt_coverage
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    basename = os.path.basename(filepath)
    model_id = basename.replace('_metrics.csv', '').replace('.csv', '')
    
    required_columns = {'iou', 'dice', 'count_error', 'coverage_error'}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {filepath}")
        
        file_columns = set(reader.fieldnames)
        missing = required_columns - file_columns
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. Found: {file_columns}"
            )
        
        rows = []
        numeric_columns = {'iou', 'dice', 'count_error', 'coverage_error',
                           'pred_count', 'gt_count', 'pred_coverage', 'gt_coverage'}
        
        for row in reader:
            converted = {}
            for k, v in row.items():
                if k in numeric_columns:
                    try:
                        converted[k] = float(v)
                    except (ValueError, TypeError):
                        converted[k] = 0.0
                else:
                    converted[k] = v
            rows.append(converted)
    
    return model_id, rows


def load_multiple_csvs(filepaths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load metrics from multiple CSV files."""
    results = {}
    for fp in filepaths:
        model_id, rows = load_metrics_csv(fp)
        if model_id in results:
            suffix = 2
            while f"{model_id}_{suffix}" in results:
                suffix += 1
            model_id = f"{model_id}_{suffix}"
        results[model_id] = rows
        print(f"  Loaded {len(rows)} samples from {model_id} ({fp})")
    
    return results


# =============================================================================
# Correlation Computation
# =============================================================================

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and p-value.
    Returns (0.0, 1.0) if computation fails.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    
    try:
        from scipy import stats
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except ImportError:
        r = np.corrcoef(x, y)[0, 1]
        return float(r), float('nan')
    except Exception:
        return 0.0, 1.0


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient and p-value.
    Returns (0.0, 1.0) if computation fails.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    try:
        from scipy import stats
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)
    except ImportError:
        # FIXED: Old code tried to import rankdata from scipy here, which would fail.
        # Safe fallback: return NaN p-value
        return 0.0, float('nan')
    except Exception:
        return 0.0, 1.0


def compute_correlation_table(
    samples: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute all correlation pairs for a set of samples.
    
    Pairs:
    - IoU vs Count Error
    - IoU vs Coverage Error
    - Dice vs Count Error
    - Dice vs Coverage Error
    - IoU vs Dice (expected: high positive)
    - Count Error vs Coverage Error (expected: moderate positive)
    """
    if len(samples) < 3:
        print("    Warning: Too few samples for reliable correlation (N<3)")
    
    iou = np.array([s['iou'] for s in samples])
    dice = np.array([s['dice'] for s in samples])
    count_err = np.array([s['count_error'] for s in samples])
    cov_err = np.array([s['coverage_error'] for s in samples])
    
    pairs = [
        ('IoU vs Count Error', iou, count_err),
        ('IoU vs Coverage Error', iou, cov_err),
        ('Dice vs Count Error', dice, count_err),
        ('Dice vs Coverage Error', dice, cov_err),
        ('IoU vs Dice', iou, dice),
        ('Count Error vs Coverage Error', count_err, cov_err),
    ]
    
    table = {}
    for pair_name, x, y in pairs:
        pearson_r, pearson_p = pearson_correlation(x, y)
        spearman_r, spearman_p = spearman_correlation(x, y)
        
        table[pair_name] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n': len(x),
        }
    
    return table


# =============================================================================
# Interpretation Helpers
# =============================================================================

def interpret_correlation(r: float) -> str:
    """Interpret correlation strength."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "weak"
    elif abs_r < 0.5:
        return "moderate"
    elif abs_r < 0.7:
        return "strong"
    else:
        return "very strong"


def format_p_value(p: float) -> str:
    """Format p-value with significance indicator."""
    if np.isnan(p):
        return "p=N/A"
    elif p < 0.001:
        return f"p<0.001***"
    elif p < 0.01:
        return f"p={p:.4f}**"
    elif p < 0.05:
        return f"p={p:.4f}*"
    else:
        return f"p={p:.4f}"


# =============================================================================
# Visualization
# =============================================================================

def generate_scatter_plots(
    samples: List[Dict[str, Any]],
    model_id: str,
    output_dir: str,
) -> List[str]:
    """Generate scatter plots for key correlation pairs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    iou = np.array([s['iou'] for s in samples])
    dice = np.array([s['dice'] for s in samples])
    count_err = np.array([s['count_error'] for s in samples])
    cov_err = np.array([s['coverage_error'] for s in samples])
    
    plots = [
        ('IoU vs Count Error', iou, count_err, 'IoU', 'Count Error'),
        ('IoU vs Coverage Error', iou, cov_err, 'IoU', 'Coverage Error'),
        ('Dice vs Count Error', dice, count_err, 'Dice', 'Count Error'),
        ('Dice vs Coverage Error', dice, cov_err, 'Dice', 'Coverage Error'),
    ]
    
    saved_paths = []
    
    for title, x, y, xlabel, ylabel in plots:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.scatter(x, y, alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
        
        if len(x) >= 3 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
            
            r, _ = pearson_correlation(x, y)
            ax.text(
                0.05, 0.95,
                f'Pearson r = {r:.3f}',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{model_id}: {title}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('vs', 'vs').lower()
        filepath = os.path.join(output_dir, f"{model_id}_{safe_title}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.append(filepath)
    
    return saved_paths


def generate_correlation_heatmap(
    corr_table: Dict[str, Dict[str, Any]],
    model_id: str,
    output_dir: str,
) -> Optional[str]:
    """Generate heatmap of all pairwise Pearson correlations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    
    metrics = ['IoU', 'Dice', 'Count Error', 'Coverage Error']
    n = len(metrics)
    matrix = np.eye(n)
    
    pair_to_idx = {
        'IoU vs Dice': (0, 1),
        'IoU vs Count Error': (0, 2),
        'IoU vs Coverage Error': (0, 3),
        'Dice vs Count Error': (1, 2),
        'Dice vs Coverage Error': (1, 3),
        'Count Error vs Coverage Error': (2, 3),
    }
    
    for pair_name, (i, j) in pair_to_idx.items():
        if pair_name in corr_table:
            r = corr_table[pair_name]['pearson_r']
            matrix[i, j] = r
            matrix[j, i] = r
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_title(f'{model_id}: Correlation Matrix', fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Pearson r')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"{model_id}_correlation_heatmap.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filepath


# =============================================================================
# Results Export
# =============================================================================

def save_correlation_csv(
    corr_table: Dict[str, Dict[str, Any]],
    model_id: str,
    filepath: str,
) -> None:
    """Save correlation table to CSV."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    fieldnames = [
        'pair', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'n', 'interpretation_pearson',
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pair_name, stats in corr_table.items():
            writer.writerow({
                'pair': pair_name,
                'pearson_r': f"{stats['pearson_r']:.4f}",
                'pearson_p': f"{stats['pearson_p']:.6f}" if not np.isnan(stats['pearson_p']) else 'N/A',
                'spearman_r': f"{stats['spearman_r']:.4f}",
                'spearman_p': f"{stats['spearman_p']:.6f}" if not np.isnan(stats['spearman_p']) else 'N/A',
                'n': stats['n'],
                'interpretation_pearson': interpret_correlation(stats['pearson_r']),
            })
    
    print(f"  Correlation table saved to: {filepath}")


# =============================================================================
# Printing
# =============================================================================

def print_correlation_table(
    model_id: str,
    corr_table: Dict[str, Dict[str, Any]],
) -> None:
    """Print formatted correlation table to console."""
    print(f"\n  {'='*70}")
    print(f"  Correlation Analysis: {model_id} (N={list(corr_table.values())[0]['n'] if corr_table else 0})")
    print(f"  {'='*70}")
    print(f"  {'Pair':<30} {'Pearson r':>10} {'Spearman r':>11} {'Interpret':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*11} {'-'*12}")
    
    for pair_name, stats in corr_table.items():
        pearson_str = f"{stats['pearson_r']:+.4f}"
        spearman_str = f"{stats['spearman_r']:+.4f}"
        interp = interpret_correlation(stats['pearson_r'])
        
        if 'IoU vs Count Error' in pair_name:
            marker = " ◄ KEY"
        elif 'Dice vs Count Error' in pair_name:
            marker = " ◄ KEY"
        else:
            marker = ""
        
        print(f"  {pair_name:<30} {pearson_str:>10} {spearman_str:>11} {interp:>12}{marker}")
    
    print(f"  {'='*70}")


def print_key_finding(corr_table: Dict[str, Dict[str, Any]]) -> None:
    """Print the key research finding with tiered scientific interpretation."""
    if 'IoU vs Count Error' not in corr_table:
        return
    
    stats = corr_table['IoU vs Count Error']
    r = stats['pearson_r']
    interp = interpret_correlation(r)
    
    print(f"\n  ╔{'='*68}╗")
    print(f"  ║  KEY RESEARCH FINDING{' '*46}║")
    print(f"  ╠{'='*68}╣")
    print(f"  ║{' '*68}║")
    print(f"  ║  Correlation(IoU, Count Error) = {r:+.4f} ({interp}){' '*24}║")
    print(f"  ║{' '*68}║")
    
    # Tiered scientific interpretation
    if abs(r) < 0.3:
        print(f"  ║  → WEAK evidence: High IoU is a poor predictor of      ║")
        print(f"  ║    count error. Segmentation quality alone is highly     ║")
        print(f"  ║    insufficient for downstream reliability.              ║")
    elif abs(r) < 0.5:
        print(f"  ║  → MODERATE evidence: IoU explains some variance in    ║")
        print(f"  ║    count error, but substantial unreliability remains.  ║")
    elif abs(r) < 0.7:
        print(f"  ║  → STRONG evidence: IoU is a significant predictor of   ║")
        print(f"  ║    count error, though not perfectly deterministic.      ║")
    else:
        print(f"  ║  → VERY STRONG evidence: IoU reliably predicts count    ║")
        print(f"  ║    error.                                            ║")
    
    print(f"  ║{' '*68}║")
    print(f"  ╚{'='*68}╝")


def print_comparison_summary(
    all_results: Dict[str, Dict[str, Any]],
) -> None:
    """Print comparison across multiple models."""
    if len(all_results) <= 1:
        return
    
    print(f"\n  {'='*80}")
    print(f"  Cross-Model Comparison: IoU vs Count Error Correlation")
    print(f"  {'='*80}")
    print(f"  {'Model':<30} {'Pearson r':>10} {'Spearman r':>11} {'N':>6}")
    print(f"  {'-'*30} {'-'*10} {'-'*11} {'-'*6}")
    
    for model_id, corr_table in all_results.items():
        if 'IoU vs Count Error' in corr_table:
            stats = corr_table['IoU vs Count Error']
            print(
                f"  {model_id:<30} "
                f"{stats['pearson_r']:>+10.4f} "
                f"{stats['spearman_r']:>+11.4f} "
                f"{stats['n']:>6}"
            )
    
    print(f"  {'='*80}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reliability Analysis: Segmentation Quality vs Downstream Metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input (CSV only - inference mode removed to prevent pipeline drift)
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--csv',
        type=str,
        nargs='+',
        required=True,
        help='One or more CSV files from evaluate.py'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_dir',
        type=str,
        default='analysis',
        help='Directory for analysis outputs'
    )
    output_group.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    print("=" * 80)
    print("Building Footprint Segmentation - Reliability Analysis")
    print("=" * 80)
    
    # Load CSVs
    print(f"\n--- Loading CSV Files ---")
    all_data = load_multiple_csvs(args.csv)
    
    # Compute correlations
    print(f"\n--- Computing Correlations ---")
    
    all_corr_tables = {}
    
    for model_id, samples in all_data.items():
        if len(samples) < 3:
            print(f"  {model_id}: Skipping (only {len(samples)} samples)")
            continue
        
        corr_table = compute_correlation_table(samples)
        all_corr_tables[model_id] = corr_table
        
        print_correlation_table(model_id, corr_table)
        print_key_finding(corr_table)
    
    # Cross-model comparison
    if len(all_corr_tables) > 1:
        print_comparison_summary(all_corr_tables)
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    for model_id, corr_table in all_corr_tables.items():
        csv_path = os.path.join(args.output_dir, f"{model_id}_correlations.csv")
        save_correlation_csv(corr_table, model_id, csv_path)
        
        if not args.no_plots:
            samples = all_data[model_id]
            plots = generate_scatter_plots(samples, model_id, args.output_dir)
            if plots:
                print(f"  Saved {len(plots)} scatter plots to {args.output_dir}/")
            
            heatmap_path = generate_correlation_heatmap(corr_table, model_id, args.output_dir)
            if heatmap_path:
                print(f"  Saved heatmap: {heatmap_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"  Models analyzed: {len(all_corr_tables)}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()