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

    # Run inference + analysis in one step
    python analyze_predictions.py --model deeplabv3plus --checkpoint checkpoints/deeplabv3plus_resnet34_best.pth --data_dir data/val

    # Custom output
    python analyze_predictions.py --csv results/model_metrics.csv --output_dir analysis --no_plots
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# CSV Loading
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
    
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    # Derive model identifier from filename
    basename = os.path.basename(filepath)
    model_id = basename.replace('_metrics.csv', '').replace('.csv', '')
    
    required_columns = {'iou', 'dice', 'count_error', 'coverage_error'}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        # Validate columns
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {filepath}")
        
        file_columns = set(reader.fieldnames)
        missing = required_columns - file_columns
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Found columns: {file_columns}"
            )
        
        # Read rows, converting numeric strings to floats
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
    """
    Load metrics from multiple CSV files.
    
    Args:
        filepaths: List of CSV file paths
    
    Returns:
        Dict mapping model_id to list of per-sample dicts
    """
    results = {}
    for fp in filepaths:
        model_id, rows = load_metrics_csv(fp)
        # Handle duplicate model IDs
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
    
    Args:
        x: 1D array
        y: 1D array (same length as x)
    
    Returns:
        Tuple of (correlation_coefficient, p_value)
        Returns (0.0, 1.0) if computation fails (e.g., zero variance)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    # Check for zero variance
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    
    try:
        from scipy import stats
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except ImportError:
        # Fallback: manual Pearson without p-value
        r = np.corrcoef(x, y)[0, 1]
        return float(r), float('nan')
    except Exception:
        return 0.0, 1.0


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient and p-value.
    
    Args:
        x: 1D array
        y: 1D array (same length as x)
    
    Returns:
        Tuple of (correlation_coefficient, p_value)
        Returns (0.0, 1.0) if computation fails
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    try:
        from scipy import stats
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)
    except ImportError:
        # Fallback: convert to ranks and use Pearson
        from scipy.stats import rankdata
        try:
            rx = rankdata(x)
            ry = rankdata(y)
            r = np.corrcoef(rx, ry)[0, 1]
            return float(r), float('nan')
        except Exception:
            return 0.0, 1.0
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
    
    Also computes:
    - IoU vs Dice (expected: high positive)
    - Count Error vs Coverage Error (expected: moderate positive)
    
    Args:
        samples: List of per-sample metric dicts
    
    Returns:
        Nested dict with correlation results
    """
    if len(samples) < 3:
        print("    Warning: Too few samples for reliable correlation (N<3)")
    
    # Extract arrays
    iou = np.array([s['iou'] for s in samples])
    dice = np.array([s['dice'] for s in samples])
    count_err = np.array([s['count_error'] for s in samples])
    cov_err = np.array([s['coverage_error'] for s in samples])
    
    # Define pairs to analyze
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
    """
    Interpret correlation strength.
    
    Args:
        r: Correlation coefficient in [-1, 1]
    
    Returns:
        Descriptive string
    """
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
    """
    Generate scatter plots for key correlation pairs.
    
    Creates:
    1. IoU vs Count Error
    2. IoU vs Coverage Error
    3. Dice vs Count Error
    4. Dice vs Coverage Error
    
    Args:
        samples: List of per-sample metric dicts
        model_id: Model identifier for plot titles
        output_dir: Directory to save plots
    
    Returns:
        List of saved file paths
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract arrays
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
        
        # Add trend line
        if len(x) >= 3 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
            
            # Add correlation coefficient
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
        
        # Save
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
    """
    Generate heatmap of all pairwise Pearson correlations.
    
    Args:
        corr_table: Output from compute_correlation_table
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
    
    # Build correlation matrix for 4 metrics
    metrics = ['IoU', 'Dice', 'Count Error', 'Coverage Error']
    n = len(metrics)
    matrix = np.eye(n)
    
    # Fill from table
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
    
    # Add text annotations
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
    """
    Save correlation table to CSV.
    
    Args:
        corr_table: Output from compute_correlation_table
        model_id: Model identifier
        filepath: Output CSV path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    fieldnames = [
        'pair',
        'pearson_r',
        'pearson_p',
        'spearman_r',
        'spearman_p',
        'n',
        'interpretation_pearson',
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
        
        # Highlight key finding
        if 'IoU vs Count Error' in pair_name:
            marker = " ◄ KEY"
        elif 'Dice vs Count Error' in pair_name:
            marker = " ◄ KEY"
        else:
            marker = ""
        
        print(f"  {pair_name:<30} {pearson_str:>10} {spearman_str:>11} {interp:>12}{marker}")
    
    print(f"  {'='*70}")


def print_key_finding(corr_table: Dict[str, Dict[str, Any]]) -> None:
    """Print the key research finding about IoU-Count Error correlation."""
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
    
    if abs(r) < 0.5:
        print(f"  ║  → High IoU does NOT reliably predict low count error.  ║")
        print(f"  ║  → Segmentation quality alone is insufficient for       ║")
        print(f"  ║    downstream urban planning reliability.               ║")
    else:
        print(f"  ║  → IoU shows {'strong' if abs(r) >= 0.7 else 'moderate'} correlation with count error.       ║")
    
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
# Inference Mode (when no CSV provided)
# =============================================================================

def run_inference_and_analyze(
    model_name: str,
    encoder_name: str,
    checkpoint_path: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    threshold: float,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Run model inference and compute per-sample metrics.
    
    This allows analyze_predictions.py to work without a pre-exported CSV.
    Imports from model.py for model construction.
    """
    from model import get_model, clean_state_dict
    
    # Import dataset class (same as evaluate.py)
    # We inline it here to avoid circular imports
    class BuildingFootprintDataset(torch.utils.data.Dataset):
        VALID_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        
        def __init__(self, image_dir: str, mask_dir: str, image_size: int = 512):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.image_size = image_size
            self.image_files = sorted([
                f for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_IMAGE_EXTS
            ])
        
        def _get_mask_path(self, img_file: str) -> Optional[str]:
            img_stem = os.path.splitext(img_file)[0]
            for ext in self.VALID_IMAGE_EXTS:
                mask_path = os.path.join(self.mask_dir, img_stem + ext)
                if os.path.exists(mask_path):
                    return mask_path
            return os.path.join(self.mask_dir, img_stem + '.png')
        
        def __len__(self) -> int:
            return len(self.image_files)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
            from PIL import Image
            import numpy as np
            
            img_file = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = self._get_mask_path(img_file)
            
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            image = torch.from_numpy(arr.transpose(2, 0, 1))
            
            mask = Image.open(mask_path).convert('L')
            arr = np.array(mask, dtype=np.float32)
            arr = (arr > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(arr).unsqueeze(0)
            
            if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode='nearest'
                ).squeeze(0)
            
            return image, mask_tensor, img_file
    
    # Detect directories
    if os.path.isdir(os.path.join(data_dir, 'images')):
        img_dir = os.path.join(data_dir, 'images')
        mask_dir = os.path.join(data_dir, 'masks')
    else:
        parent = os.path.dirname(data_dir)
        img_dir = data_dir
        mask_dir = os.path.join(parent, 'masks')
    
    # Create model
    model = get_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = clean_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict, strict=True)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    dataset = BuildingFootprintDataset(img_dir, mask_dir, image_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    # Import metric functions from evaluate.py if possible, else inline
    try:
        from evaluate import compute_iou, compute_dice, compute_count_error, compute_coverage_error
    except ImportError:
        # Inline implementations
        def compute_iou(pred, gt):
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            return 1.0 if union == 0 else float(intersection) / float(union)
        
        def compute_dice(pred, gt):
            intersection = np.logical_and(pred, gt).sum()
            total = pred.sum() + gt.sum()
            return 1.0 if total == 0 else 2.0 * float(intersection) / float(total)
        
        def count_buildings(mask):
            if mask.sum() == 0:
                return 0
            try:
                from scipy import ndimage
                _, n = ndimage.label(mask)
                return n
            except ImportError:
                return 0
        
        def compute_count_error(pred, gt):
            return abs(count_buildings(pred) - count_buildings(gt)) / max(count_buildings(gt), 1)
        
        def compute_coverage_error(pred, gt):
            pc = pred.sum() / pred.size
            gc = gt.sum() / gt.size
            return abs(pc - gc) / max(gc, 1e-6)
    
    # Run inference
    per_sample = []
    
    with torch.no_grad():
        for images, masks, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            preds = (torch.sigmoid(outputs) > threshold).float().cpu().numpy()
            masks_np = masks.numpy()
            
            for i in range(preds.shape[0]):
                pred_mask = preds[i, 0]
                gt_mask = masks_np[i, 0]
                fname = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
                
                per_sample.append({
                    'filename': fname,
                    'iou': compute_iou(pred_mask, gt_mask),
                    'dice': compute_dice(pred_mask, gt_mask),
                    'count_error': compute_count_error(pred_mask, gt_mask),
                    'coverage_error': compute_coverage_error(pred_mask, gt_mask),
                    'pred_count': 0,  # Simplified for inference mode
                    'gt_count': 0,
                    'pred_coverage': pred_mask.sum() / pred_mask.size,
                    'gt_coverage': gt_mask.sum() / gt_mask.size,
                })
    
    # Aggregate
    aggregate = {
        'iou': np.mean([s['iou'] for s in per_sample]),
        'dice': np.mean([s['dice'] for s in per_sample]),
        'count_error': np.mean([s['count_error'] for s in per_sample]),
        'coverage_error': np.mean([s['coverage_error'] for s in per_sample]),
    }
    
    return aggregate, per_sample


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reliability Analysis: Segmentation Quality vs Downstream Metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input mode: CSV vs inference
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--csv',
        type=str,
        nargs='+',
        default=None,
        help='One or more CSV files from evaluate.py (skip inference)'
    )
    
    # Inference mode arguments (used if --csv not provided)
    infer_group = parser.add_argument_group('Inference (if --csv not specified)')
    infer_group.add_argument(
        '--model',
        type=str,
        default=None,
        choices=SUPPORTED_MODELS if 'SUPPORTED_MODELS' in dir() else None,
        help='Model architecture for inference mode'
    )
    infer_group.add_argument(
        '--encoder',
        type=str,
        default='resnet34',
        help='Encoder for inference mode'
    )
    infer_group.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path for inference mode'
    )
    infer_group.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Data directory for inference mode'
    )
    infer_group.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='Image size for inference mode'
    )
    infer_group.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    infer_group.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader workers for inference'
    )
    infer_group.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Binarization threshold'
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
    
    # System
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute device'
    )
    
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    # Import SUPPORTED_MODELS for choices validation
    from model import SUPPORTED_MODELS
    
    args = parse_args()
    
    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    print("=" * 80)
    print("Building Footprint Segmentation - Reliability Analysis")
    print("=" * 80)
    
    # ------------------------------------------------------------------
    # Mode 1: CSV analysis
    # ------------------------------------------------------------------
    if args.csv is not None:
        print(f"\n--- Loading CSV Files ---")
        all_data = load_multiple_csvs(args.csv)
        
    # ------------------------------------------------------------------
    # Mode 2: Inference + analysis
    # ------------------------------------------------------------------
    else:
        if args.model is None or args.checkpoint is None or args.data_dir is None:
            raise ValueError(
                "Either --csv or all of --model, --checkpoint, --data_dir "
                "must be specified."
            )
        
        print(f"\n--- Running Inference ---")
        print(f"  Model: {args.model}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Data: {args.data_dir}")
        
        # Get encoder/image_size from checkpoint if possible
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        encoder_name = ckpt.get('encoder_name', args.encoder)
        image_size = ckpt.get('image_size', args.image_size)
        
        aggregate, per_sample = run_inference_and_analyze(
            model_name=args.model,
            encoder_name=encoder_name,
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            device=device,
        )
        
        model_id = f"{args.model}_{encoder_name}"
        all_data = {model_id: per_sample}
        
        print(f"\n  Aggregate metrics:")
        print(f"    IoU: {aggregate['iou']:.4f}")
        print(f"    Dice: {aggregate['dice']:.4f}")
        print(f"    Count Error: {aggregate['count_error']:.4f}")
        print(f"    Coverage Error: {aggregate['coverage_error']:.4f}")
    
    # ------------------------------------------------------------------
    # Compute correlations for each model
    # ------------------------------------------------------------------
    print(f"\n--- Computing Correlations ---")
    
    all_corr_tables = {}
    
    for model_id, samples in all_data.items():
        if len(samples) < 3:
            print(f"  {model_id}: Skipping (only {len(samples)} samples)")
            continue
        
        corr_table = compute_correlation_table(samples)
        all_corr_tables[model_id] = corr_table
        
        # Print table
        print_correlation_table(model_id, corr_table)
        
        # Print key finding
        print_key_finding(corr_table)
    
    # ------------------------------------------------------------------
    # Cross-model comparison
    # ------------------------------------------------------------------
    if len(all_corr_tables) > 1:
        print_comparison_summary(all_corr_tables)
    
    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    
    for model_id, corr_table in all_corr_tables.items():
        # Save correlation CSV
        csv_path = os.path.join(args.output_dir, f"{model_id}_correlations.csv")
        save_correlation_csv(corr_table, model_id, csv_path)
        
        # Generate plots
        if not args.no_plots:
            samples = all_data[model_id]
            plots = generate_scatter_plots(samples, model_id, args.output_dir)
            if plots:
                print(f"  Saved {len(plots)} scatter plots to {args.output_dir}/")
            
            heatmap_path = generate_correlation_heatmap(corr_table, model_id, args.output_dir)
            if heatmap_path:
                print(f"  Saved heatmap: {heatmap_path}")
    
    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"  Models analyzed: {len(all_corr_tables)}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()