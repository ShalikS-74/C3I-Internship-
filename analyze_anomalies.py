"""
analyze_anomalies.py - Anomaly Detection for Segmentation Reliability

Identifies samples where segmentation quality (IoU/Dice) and downstream
reliability metrics (Count Error, Coverage Error) are inconsistent.

Anomaly Categories:
- False Reliable:  High IoU but High Count Error (dangerous: looks good, isn't)
- Hidden Good:     Low IoU but Low Count Error (underestimated quality)
- Coverage Mismatch: High IoU but High Coverage Error (spatial distortion)
- Statistical Outlier: >2std from mean on any metric

Usage:
    python analyze_anomalies.py --csv results/deeplabv3plus_resnet34_metrics.csv
    python analyze_anomalies.py --csv results/*.csv --output_dir anomalies
    python analyze_anomalies.py --csv results/model.csv --zscore_threshold 1.5 --no_plots
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np


# =============================================================================
# CSV Loading (same format as analyze_predictions.py)
# =============================================================================

def load_metrics_csv(filepath: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load per-sample metrics from CSV exported by evaluate.py.
    
    Expected columns: filename, iou, dice, count_error, coverage_error,
                      pred_count, gt_count, pred_coverage, gt_coverage
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Tuple of (model_identifier, list of per-sample dicts)
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
                f"CSV missing required columns: {missing}. "
                f"Found: {file_columns}"
            )
        
        rows = []
        numeric_columns = {
            'iou', 'dice', 'count_error', 'coverage_error',
            'pred_count', 'gt_count', 'pred_coverage', 'gt_coverage'
        }
        
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


# =============================================================================
# Anomaly Classification
# =============================================================================

class AnomalyClassifier:
    """
    Classifies samples into anomaly categories based on metric inconsistencies.
    
    Thresholds:
    - Quartile-based: "high" = above Q3, "low" = below Q1
    - Z-score-based: outlier = |z| > threshold (default 2.0)
    - Custom: user-specified absolute thresholds
    """
    
    def __init__(
        self,
        zscore_threshold: float = 2.0,
        use_quartiles: bool = True,
        use_zscore: bool = True,
        custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Args:
            zscore_threshold: Z-score threshold for outlier detection
            use_quartiles: Use quartile-based anomaly classification
            use_zscore: Use z-score-based outlier detection
            custom_thresholds: Dict of custom thresholds, e.g.
                {"iou": {"high": 0.8, "low": 0.5}}
        """
        self.zscore_threshold = zscore_threshold
        self.use_quartiles = use_quartiles
        self.use_zscore = use_zscore
        self.custom_thresholds = custom_thresholds or {}
        
        # Computed from data
        self.medians = {}
        self.q1 = {}
        self.q3 = {}
        self.means = {}
        self.stds = {}
    
    def fit(self, samples: List[Dict[str, Any]]) -> None:
        """
        Compute thresholds from data distribution.
        
        Args:
            samples: List of per-sample metric dicts
        """
        metric_keys = ['iou', 'dice', 'count_error', 'coverage_error']
        
        for key in metric_keys:
            values = np.array([s[key] for s in samples])
            
            self.medians[key] = float(np.median(values))
            self.q1[key] = float(np.percentile(values, 25))
            self.q3[key] = float(np.percentile(values, 75))
            self.means[key] = float(np.mean(values))
            self.stds[key] = float(np.std(values))
            
            # Prevent division by zero
            if self.stds[key] < 1e-10:
                self.stds[key] = 1.0
    
    def _is_high_quartile(self, key: str, value: float) -> bool:
        """Check if value is above Q3 (high)."""
        if key in self.custom_thresholds and 'high' in self.custom_thresholds[key]:
            return value > self.custom_thresholds[key]['high']
        return value > self.q3[key]
    
    def _is_low_quartile(self, key: str, value: float) -> bool:
        """Check if value is below Q1 (low)."""
        if key in self.custom_thresholds and 'low' in self.custom_thresholds[key]:
            return value < self.custom_thresholds[key]['low']
        return value < self.q1[key]
    
    def _is_zscore_outlier(self, key: str, value: float) -> bool:
        """Check if value is a statistical outlier by z-score."""
        if not self.use_zscore:
            return False
        z = abs(value - self.means[key]) / self.stds[key]
        return z > self.zscore_threshold
    
    def classify_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single sample into anomaly categories.
        
        Returns dict with:
        - anomaly_type: Primary anomaly category (or "normal")
        - anomaly_flags: Set of all applicable flags
        - zscores: Dict of z-scores for each metric
        - quartile_positions: "high", "low", or "mid" for each metric
        """
        flags = set()
        zscores = {}
        quartile_positions = {}
        
        metric_keys = ['iou', 'dice', 'count_error', 'coverage_error']
        
        for key in metric_keys:
            value = sample[key]
            
            # Z-score
            z = (value - self.means[key]) / self.stds[key]
            zscores[key] = z
            
            if self._is_zscore_outlier(key, value):
                flags.add(f"zscore_outlier_{key}")
            
            # Quartile position
            if self.use_quartiles:
                if self._is_high_quartile(key, value):
                    quartile_positions[key] = "high"
                elif self._is_low_quartile(key, value):
                    quartile_positions[key] = "low"
                else:
                    quartile_positions[key] = "mid"
            else:
                quartile_positions[key] = "unknown"
        
        # --- Anomaly Category Classification ---
        
        # False Reliable: High IoU + High Count Error
        # This is the DANGEROUS case: segmentation looks good but count is wrong
        if (quartile_positions.get('iou') == 'high' and 
            quartile_positions.get('count_error') == 'high'):
            flags.add('false_reliable')
        
        # Also catch with Dice as secondary indicator
        if (quartile_positions.get('dice') == 'high' and 
            quartile_positions.get('count_error') == 'high' and
            'false_reliable' not in flags):
            flags.add('false_reliable_dice')
        
        # Hidden Good: Low IoU + Low Count Error
        # Model looks bad but actually gets count right
        if (quartile_positions.get('iou') == 'low' and 
            quartile_positions.get('count_error') == 'low'):
            flags.add('hidden_good')
        
        # Coverage Mismatch: High IoU + High Coverage Error
        # Good segmentation overlap but wrong total area
        if (quartile_positions.get('iou') == 'high' and 
            quartile_positions.get('coverage_error') == 'high'):
            flags.add('coverage_mismatch')
        
        # Low Coverage Error + High Count Error (or vice versa)
        # Good area but wrong number of buildings
        if (quartile_positions.get('coverage_error') == 'low' and 
            quartile_positions.get('count_error') == 'high'):
            flags.add('count_area_dissociation')
        
        # High Coverage Error + Low Count Error
        # Wrong area but right number of buildings
        if (quartile_positions.get('coverage_error') == 'high' and 
            quartile_positions.get('count_error') == 'low'):
            flags.add('area_count_dissociation')
        
        # Perfect: High IoU + Low Count Error + Low Coverage Error
        if (quartile_positions.get('iou') == 'high' and 
            quartile_positions.get('count_error') == 'low' and
            quartile_positions.get('coverage_error') == 'low'):
            flags.add('perfect_reliable')
        
        # Terrible: Low IoU + High Count Error + High Coverage Error
        if (quartile_positions.get('iou') == 'low' and 
            quartile_positions.get('count_error') == 'high' and
            quartile_positions.get('coverage_error') == 'high'):
            flags.add('terrible')
        
        # Determine primary anomaly type
        if 'false_reliable' in flags:
            primary_type = 'FALSE_RELIABLE'
        elif 'false_reliable_dice' in flags:
            primary_type = 'FALSE_RELIABLE_DICE'
        elif 'hidden_good' in flags:
            primary_type = 'HIDDEN_GOOD'
        elif 'coverage_mismatch' in flags:
            primary_type = 'COVERAGE_MISMATCH'
        elif 'count_area_dissociation' in flags:
            primary_type = 'COUNT_AREA_DISSOCIATION'
        elif 'area_count_dissociation' in flags:
            primary_type = 'AREA_COUNT_DISSOCIATION'
        elif 'perfect_reliable' in flags:
            primary_type = 'PERFECT_RELIABLE'
        elif 'terrible' in flags:
            primary_type = 'TERRIBLE'
        elif len([f for f in flags if f.startswith('zscore_outlier_')]) > 0:
            primary_type = 'STATISTICAL_OUTLIER'
        else:
            primary_type = 'NORMAL'
        
        return {
            'anomaly_type': primary_type,
            'anomaly_flags': flags,
            'zscores': zscores,
            'quartile_positions': quartile_positions,
        }


# =============================================================================
# Anomaly Analysis
# =============================================================================

def analyze_anomalies(
    samples: List[Dict[str, Any]],
    classifier: AnomalyClassifier,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run anomaly detection on all samples.
    
    Args:
        samples: List of per-sample metric dicts
        classifier: Fitted AnomalyClassifier
    
    Returns:
        Tuple of:
        - annotated_samples: Samples with anomaly fields added
        - summary: Aggregate anomaly statistics
    """
    classifier.fit(samples)
    
    annotated = []
    type_counts = defaultdict(int)
    flag_counts = defaultdict(int)
    
    for sample in samples:
        result = classifier.classify_sample(sample)
        
        # Merge anomaly info into sample
        annotated_sample = dict(sample)
        annotated_sample['anomaly_type'] = result['anomaly_type']
        annotated_sample['anomaly_flags'] = '|'.join(sorted(result['anomaly_flags'])) if result['anomaly_flags'] else 'none'
        annotated_sample['zscore_iou'] = result['zscores']['iou']
        annotated_sample['zscore_dice'] = result['zscores']['dice']
        annotated_sample['zscore_count_error'] = result['zscores']['count_error']
        annotated_sample['zscore_coverage_error'] = result['zscores']['coverage_error']
        
        annotated.append(annotated_sample)
        
        # Count
        type_counts[result['anomaly_type']] += 1
        for flag in result['anomaly_flags']:
            flag_counts[flag] += 1
    
    # Build summary
    n = len(samples)
    summary = {
        'total_samples': n,
        'anomaly_type_counts': dict(type_counts),
        'anomaly_type_percentages': {k: v / n * 100 for k, v in type_counts.items()},
        'flag_counts': dict(flag_counts),
        'thresholds': {
            'iou': {'median': classifier.medians['iou'], 'q1': classifier.q1['iou'], 'q3': classifier.q3['iou']},
            'dice': {'median': classifier.medians['dice'], 'q1': classifier.q1['dice'], 'q3': classifier.q3['dice']},
            'count_error': {'median': classifier.medians['count_error'], 'q1': classifier.q1['count_error'], 'q3': classifier.q3['count_error']},
            'coverage_error': {'median': classifier.medians['coverage_error'], 'q1': classifier.q1['coverage_error'], 'q3': classifier.q3['coverage_error']},
        },
        'zscore_threshold': classifier.zscore_threshold,
    }
    
    return annotated, summary


def get_samples_by_type(
    annotated: List[Dict[str, Any]],
    anomaly_type: str,
) -> List[Dict[str, Any]]:
    """Filter samples by anomaly type."""
    return [s for s in annotated if s['anomaly_type'] == anomaly_type]


def compute_type_statistics(
    annotated: List[Dict[str, Any]],
    anomaly_type: str,
) -> Dict[str, float]:
    """
    Compute mean metrics for samples of a given anomaly type.
    
    Args:
        annotated: Annotated sample list
        anomaly_type: Type to filter by
    
    Returns:
        Dict with mean iou, dice, count_error, coverage_error
    """
    filtered = get_samples_by_type(annotated, anomaly_type)
    
    if not filtered:
        return {
            'count': 0, 'iou': 0.0, 'dice': 0.0,
            'count_error': 0.0, 'coverage_error': 0.0,
        }
    
    return {
        'count': len(filtered),
        'iou': np.mean([s['iou'] for s in filtered]),
        'dice': np.mean([s['dice'] for s in filtered]),
        'count_error': np.mean([s['count_error'] for s in filtered]),
        'coverage_error': np.mean([s['coverage_error'] for s in filtered]),
    }


# =============================================================================
# Printing
# =============================================================================

def print_thresholds(summary: Dict[str, Any]) -> None:
    """Print computed thresholds."""
    print(f"\n  {'─'*50}")
    print(f"  Computed Thresholds (Quartile-Based)")
    print(f"  {'─'*50}")
    print(f"  {'Metric':<20} {'Q1':>8} {'Median':>8} {'Q3':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    
    for metric in ['iou', 'dice', 'count_error', 'coverage_error']:
        t = summary['thresholds'][metric]
        print(f"  {metric:<20} {t['q1']:>8.4f} {t['median']:>8.4f} {t['q3']:>8.4f}")
    
    print(f"  {'─'*50}")
    print(f"  Z-score outlier threshold: {summary['zscore_threshold']}")
    print(f"  {'─'*50}")


def print_anomaly_summary(
    model_id: str,
    summary: Dict[str, Any],
    annotated: List[Dict[str, Any]],
) -> None:
    """Print anomaly distribution summary."""
    n = summary['total_samples']
    
    print(f"\n  {'═'*60}")
    print(f"  Anomaly Distribution: {model_id} (N={n})")
    print(f"  {'═'*60}")
    print(f"  {'Type':<30} {'Count':>6} {'%':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*8}")
    
    # Sort by count descending
    type_counts = summary['anomaly_type_counts']
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
    
    for atype, count in sorted_types:
        pct = summary['anomaly_type_percentages'][atype]
        marker = ""
        
        if atype == 'FALSE_RELIABLE':
            marker = " ◄ DANGEROUS"
        elif atype == 'HIDDEN_GOOD':
            marker = " ◄ NOTEWORTHY"
        elif atype == 'NORMAL':
            marker = ""
        elif atype == 'PERFECT_RELIABLE':
            marker = " ✓"
        elif atype == 'TERRIBLE':
            marker = " ✗"
        
        print(f"  {atype:<30} {count:>6} {pct:>7.1f}%{marker}")
    
    print(f"  {'═'*60}")
    
    # Print detailed stats for key anomaly types
    key_types = ['FALSE_RELIABLE', 'HIDDEN_GOOD', 'COVERAGE_MISMATCH', 
                 'PERFECT_RELIABLE', 'TERRIBLE']
    
    print(f"\n  {'─'*60}")
    print(f"  Mean Metrics by Anomaly Type")
    print(f"  {'─'*60}")
    print(f"  {'Type':<25} {'N':>4} {'IoU':>7} {'Dice':>7} {'CntErr':>7} {'CovErr':>7}")
    print(f"  {'─'*25} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    
    for atype in key_types:
        stats = compute_type_statistics(annotated, atype)
        if stats['count'] > 0:
            print(
                f"  {atype:<25} "
                f"{stats['count']:>4} "
                f"{stats['iou']:>7.4f} "
                f"{stats['dice']:>7.4f} "
                f"{stats['count_error']:>7.4f} "
                f"{stats['coverage_error']:>7.4f}"
            )
    
    print(f"  {'─'*60}")


def print_false_reliable_details(annotated: List[Dict[str, Any]]) -> None:
    """Print detailed info about False Reliable samples (most dangerous)."""
    false_reliable = get_samples_by_type(annotated, 'FALSE_RELIABLE')
    
    if not false_reliable:
        return
    
    print(f"\n  {'═'*60}")
    print(f"  FALSE RELIABLE SAMPLES (High IoU, High Count Error)")
    print(f"  These samples appear well-segmented but have wrong building counts!")
    print(f"  {'═'*60}")
    print(f"  {'Filename':<30} {'IoU':>7} {'Dice':>7} {'CntErr':>7} {'CovErr':>7}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    
    # Sort by count_error descending (worst first)
    false_reliable_sorted = sorted(false_reliable, key=lambda x: -x['count_error'])
    
    for s in false_reliable_sorted[:20]:  # Show top 20 worst
        print(
            f"  {s['filename']:<30} "
            f"{s['iou']:>7.4f} "
            f"{s['dice']:>7.4f} "
            f"{s['count_error']:>7.4f} "
            f"{s['coverage_error']:>7.4f}"
        )
    
    if len(false_reliable) > 20:
        print(f"  ... and {len(false_reliable) - 20} more")
    
    print(f"  {'═'*60}")


def print_hidden_good_details(annotated: List[Dict[str, Any]]) -> None:
    """Print detailed info about Hidden Good samples."""
    hidden_good = get_samples_by_type(annotated, 'HIDDEN_GOOD')
    
    if not hidden_good:
        return
    
    print(f"\n  {'═'*60}")
    print(f"  HIDDEN GOOD SAMPLES (Low IoU, Low Count Error)")
    print(f"  These samples have poor IoU but correct building counts!")
    print(f"  {'═'*60}")
    print(f"  {'Filename':<30} {'IoU':>7} {'Dice':>7} {'CntErr':>7} {'CovErr':>7}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    
    hidden_good_sorted = sorted(hidden_good, key=lambda x: x['iou'])
    
    for s in hidden_good_sorted[:20]:
        print(
            f"  {s['filename']:<30} "
            f"{s['iou']:>7.4f} "
            f"{s['dice']:>7.4f} "
            f"{s['count_error']:>7.4f} "
            f"{s['coverage_error']:>7.4f}"
        )
    
    if len(hidden_good) > 20:
        print(f"  ... and {len(hidden_good) - 20} more")
    
    print(f"  {'═'*60}")


# =============================================================================
# Visualization
# =============================================================================

def generate_anomaly_scatter(
    annotated: List[Dict[str, Any]],
    model_id: str,
    output_dir: str,
) -> Optional[str]:
    """
    Generate scatter plot colored by anomaly type.
    
    X-axis: IoU
    Y-axis: Count Error
    Color: Anomaly type
    
    Args:
        annotated: Annotated sample list
        model_id: Model identifier
        output_dir: Directory to save plot
    
    Returns:
        Saved file path, or None if matplotlib unavailable
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color map for anomaly types
    type_colors = {
        'NORMAL': '#cccccc',
        'FALSE_RELIABLE': '#ff4444',       # Red - dangerous
        'FALSE_RELIABLE_DICE': '#ff8800',  # Orange
        'HIDDEN_GOOD': '#44aaff',          # Blue - interesting
        'COVERAGE_MISMATCH': '#ffaa00',    # Yellow
        'COUNT_AREA_DISSOCIATION': '#cc44ff',
        'AREA_COUNT_DISSOCIATION': '#44ffaa',
        'PERFECT_RELIABLE': '#44ff44',     # Green - ideal
        'TERRIBLE': '#ff0000',             # Dark red
        'STATISTICAL_OUTLIER': '#ff66ff',  # Pink
    }
    
    # Group by type
    type_groups = defaultdict(list)
    for s in annotated:
        type_groups[s['anomaly_type']].append(s)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each type
    for atype, samples in sorted(type_groups.items()):
        ious = [s['iou'] for s in samples]
        count_errs = [s['count_error'] for s in samples]
        color = type_colors.get(atype, '#999999')
        label = f"{atype} (n={len(samples)})"
        
        ax.scatter(ious, count_errs, c=color, label=label, 
                   alpha=0.7, edgecolors='k', linewidths=0.5, s=50)
    
    # Add quadrant lines at medians
    ious_all = [s['iou'] for s in annotated]
    count_errs_all = [s['count_error'] for s in annotated]
    med_iou = np.median(ious_all)
    med_count = np.median(count_errs_all)
    
    ax.axvline(x=med_iou, color='gray', linestyle='--', alpha=0.5, label=f'Median IoU={med_iou:.3f}')
    ax.axhline(y=med_count, color='gray', linestyle=':', alpha=0.5, label=f'Median CountErr={med_count:.3f}')
    
    # Annotate quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.98, ylim[1]*0.98, 'High IoU\nHigh CountErr', 
            ha='right', va='top', fontsize=9, color='red', alpha=0.6)
    ax.text(xlim[0]*0.98+0.02, ylim[1]*0.98, 'Low IoU\nHigh CountErr', 
            ha='left', va='top', fontsize=9, color='gray', alpha=0.6)
    ax.text(xlim[0]*0.98+0.02, ylim[0]*0.98+0.02, 'Low IoU\nLow CountErr', 
            ha='left', va='bottom', fontsize=9, color='blue', alpha=0.6)
    ax.text(xlim[1]*0.98, ylim[0]*0.98+0.02, 'High IoU\nLow CountErr', 
            ha='right', va='bottom', fontsize=9, color='green', alpha=0.6)
    
    ax.set_xlabel('IoU', fontsize=12)
    ax.set_ylabel('Count Error', fontsize=12)
    ax.set_title(f'{model_id}: Anomaly Map (IoU vs Count Error)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"{model_id}_anomaly_scatter.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filepath


def generate_anomaly_bar_chart(
    summary: Dict[str, Any],
    model_id: str,
    output_dir: str,
) -> Optional[str]:
    """
    Generate bar chart of anomaly type distribution.
    
    Args:
        summary: Anomaly summary dict
        model_id: Model identifier
        output_dir: Directory to save plot
    
    Returns:
        Saved file path, or None if matplotlib unavailable
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    
    type_counts = summary['anomaly_type_counts']
    if not type_counts:
        return None
    
    # Sort by count
    sorted_items = sorted(type_counts.items(), key=lambda x: -x[1])
    types = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    # Colors
    type_colors = {
        'NORMAL': '#cccccc',
        'FALSE_RELIABLE': '#ff4444',
        'FALSE_RELIABLE_DICE': '#ff8800',
        'HIDDEN_GOOD': '#44aaff',
        'COVERAGE_MISMATCH': '#ffaa00',
        'COUNT_AREA_DISSOCIATION': '#cc44ff',
        'AREA_COUNT_DISSOCIATION': '#44ffaa',
        'PERFECT_RELIABLE': '#44ff44',
        'TERRIBLE': '#ff0000',
        'STATISTICAL_OUTLIER': '#ff66ff',
    }
    
    colors = [type_colors.get(t, '#999999') for t in types]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bars = ax.barh(types, counts, color=colors, edgecolor='k', linewidth=0.5)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title(f'{model_id}: Anomaly Type Distribution', fontsize=14)
    ax.invert_yaxis()  # Top type at top
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"{model_id}_anomaly_distribution.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filepath


# =============================================================================
# Export
# =============================================================================

def save_anomaly_csv(
    annotated: List[Dict[str, Any]],
    filepath: str,
) -> None:
    """
    Save annotated samples with anomaly classifications to CSV.
    
    Additional columns beyond evaluate.py output:
    - anomaly_type: Primary classification
    - anomaly_flags: Pipe-separated list of all flags
    - zscore_iou, zscore_dice, zscore_count_error, zscore_coverage_error
    
    Args:
        annotated: Annotated sample list
        filepath: Output CSV path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    fieldnames = [
        'filename',
        'iou', 'dice', 'count_error', 'coverage_error',
        'pred_count', 'gt_count', 'pred_coverage', 'gt_coverage',
        'anomaly_type',
        'anomaly_flags',
        'zscore_iou', 'zscore_dice', 'zscore_count_error', 'zscore_coverage_error',
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(annotated)
    
    print(f"  Annotated samples saved to: {filepath}")


def save_anomaly_summary_csv(
    model_id: str,
    summary: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Save anomaly summary statistics to CSV.
    
    Args:
        model_id: Model identifier
        summary: Summary dict from analyze_anomalies
        filepath: Output CSV path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    fieldnames = ['model_id', 'anomaly_type', 'count', 'percentage']
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for atype, count in summary['anomaly_type_counts'].items():
            writer.writerow({
                'model_id': model_id,
                'anomaly_type': atype,
                'count': count,
                'percentage': f"{summary['anomaly_type_percentages'][atype]:.2f}",
            })
    
    print(f"  Summary saved to: {filepath}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Anomaly Detection for Segmentation Reliability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--csv',
        type=str,
        nargs='+',
        required=True,
        help='One or more CSV files from evaluate.py'
    )
    
    # Threshold settings
    thresh_group = parser.add_argument_group('Thresholds')
    thresh_group.add_argument(
        '--zscore_threshold',
        type=float,
        default=2.0,
        help='Z-score threshold for statistical outlier detection'
    )
    thresh_group.add_argument(
        '--no_quartiles',
        action='store_true',
        help='Disable quartile-based classification (use only z-score)'
    )
    thresh_group.add_argument(
        '--no_zscore',
        action='store_true',
        help='Disable z-score outlier detection (use only quartiles)'
    )
    
    # Custom thresholds (advanced)
    custom_group = parser.add_argument_group('Custom Thresholds (optional)')
    custom_group.add_argument(
        '--iou_high',
        type=float,
        default=None,
        help='Custom IoU threshold for "high" (overrides Q3)'
    )
    custom_group.add_argument(
        '--iou_low',
        type=float,
        default=None,
        help='Custom IoU threshold for "low" (overrides Q1)'
    )
    custom_group.add_argument(
        '--count_error_high',
        type=float,
        default=None,
        help='Custom Count Error threshold for "high"'
    )
    custom_group.add_argument(
        '--count_error_low',
        type=float,
        default=None,
        help='Custom Count Error threshold for "low"'
    )
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_dir',
        type=str,
        default='anomalies',
        help='Directory for analysis outputs'
    )
    output_group.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip plot generation'
    )
    output_group.add_argument(
        '--max_details',
        type=int,
        default=20,
        help='Maximum number of samples to show in detail lists'
    )
    
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    print("=" * 80)
    print("Building Footprint Segmentation - Anomaly Analysis")
    print("=" * 80)
    
    # Build custom thresholds if specified
    custom_thresholds = {}
    if args.iou_high is not None or args.iou_low is not None:
        custom_thresholds['iou'] = {}
        if args.iou_high is not None:
            custom_thresholds['iou']['high'] = args.iou_high
        if args.iou_low is not None:
            custom_thresholds['iou']['low'] = args.iou_low
    if args.count_error_high is not None or args.count_error_low is not None:
        custom_thresholds['count_error'] = {}
        if args.count_error_high is not None:
            custom_thresholds['count_error']['high'] = args.count_error_high
        if args.count_error_low is not None:
            custom_thresholds['count_error']['low'] = args.count_error_low
    
    # Create classifier
    classifier = AnomalyClassifier(
        zscore_threshold=args.zscore_threshold,
        use_quartiles=not args.no_quartiles,
        use_zscore=not args.no_zscore,
        custom_thresholds=custom_thresholds if custom_thresholds else None,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each CSV
    all_summaries = {}
    
    for csv_path in args.csv:
        print(f"\n{'─'*80}")
        print(f"Processing: {csv_path}")
        print(f"{'─'*80}")
        
        # Load
        model_id, samples = load_metrics_csv(csv_path)
        print(f"  Loaded {len(samples)} samples for model: {model_id}")
        
        if len(samples) < 4:
            print(f"  WARNING: Too few samples for reliable anomaly detection (N={len(samples)})")
        
        # Analyze
        annotated, summary = analyze_anomalies(samples, classifier)
        all_summaries[model_id] = summary
        
        # Print thresholds
        print_thresholds(summary)
        
        # Print summary
        print_anomaly_summary(model_id, summary, annotated)
        
        # Print dangerous details
        print_false_reliable_details(annotated)
        print_hidden_good_details(annotated)
        
        # Save CSVs
        csv_out = os.path.join(args.output_dir, f"{model_id}_anomalies.csv")
        save_anomaly_csv(annotated, csv_out)
        
        summary_csv = os.path.join(args.output_dir, f"{model_id}_anomaly_summary.csv")
        save_anomaly_summary_csv(model_id, summary, summary_csv)
        
        # Generate plots
        if not args.no_plots:
            scatter_path = generate_anomaly_scatter(annotated, model_id, args.output_dir)
            if scatter_path:
                print(f"  Saved scatter plot: {scatter_path}")
            
            bar_path = generate_anomaly_bar_chart(summary, model_id, args.output_dir)
            if bar_path:
                print(f"  Saved bar chart: {bar_path}")
    
    # Cross-model comparison if multiple CSVs
    if len(all_summaries) > 1:
        print(f"\n{'═'*80}")
        print("Cross-Model Anomaly Comparison")
        print(f"{'═'*80}")
        
        # Compare False Reliable rates
        print(f"\n  {'Model':<30} {'N':>5} {'FALSE_REL':>9} {'HIDDEN_GOOD':>12} {'NORMAL':>7}")
        print(f"  {'─'*30} {'─'*5} {'─'*9} {'─'*12} {'─'*7}")
        
        for model_id, summary in all_summaries.items():
            n = summary['total_samples']
            fr = summary['anomaly_type_counts'].get('FALSE_RELIABLE', 0)
            hg = summary['anomaly_type_counts'].get('HIDDEN_GOOD', 0)
            nm = summary['anomaly_type_counts'].get('NORMAL', 0)
            
            print(
                f"  {model_id:<30} "
                f"{n:>5} "
                f"{fr:>5} ({fr/n*100:>3.1f}%) "
                f"{hg:>5} ({hg/n*100:>3.1f}%) "
                f"{nm:>5} ({nm/n*100:>3.1f}%)"
            )
        
        print(f"  {'═'*80}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("Anomaly Analysis Complete")
    print(f"{'='*80}")
    print(f"  Models analyzed: {len(all_summaries)}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()