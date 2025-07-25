"""
File: visualization.py
Directory: memorization_reproduction/src/

Visualization module for reproducing Morris et al. figures and creating experimental dashboards.
Provides publication-quality plots for memorization capacity findings.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from datetime import datetime
import json

# Import our modules
from experiment_runner import ExperimentSuite, ExperimentConfig
from capacity_estimator import CapacityEstimate


# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Academic figure settings
FIGURE_DPI = 300
FIGURE_SIZE = (10, 8)
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12

# Morris et al. reference values
MORRIS_BITS_PER_PARAM = 3.6
MORRIS_COLORS = {
    'target': '#e74c3c',  # Red for Morris et al. reference
    'observed': '#3498db',  # Blue for our observations
    'good_fit': '#27ae60',  # Green for good fits
    'poor_fit': '#f39c12',  # Orange for poor fits
    'background': '#ecf0f1'  # Light gray for backgrounds
}


def setup_figure_style() -> None:
    """Set up consistent figure styling for all plots."""
    plt.rcParams.update({
        'figure.figsize': FIGURE_SIZE,
        'figure.dpi': FIGURE_DPI,
        'font.size': TICK_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_memorization_vs_dataset_size(
    capacity_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Memorization vs Dataset Size (Morris et al. Figure 1 Reproduction)"
) -> plt.Figure:
    """
    Reproduce Morris et al. Figure 1: Memorization plateaus across dataset sizes.
    
    Args:
        capacity_results: Results from run_capacity_experiments
        save_path: Optional path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure object
    """
    setup_figure_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data from capacity results
    individual_results = capacity_results['individual_results']
    
    # Group by model size
    model_groups = {}
    for result in individual_results:
        model_params = result['model_params']
        if model_params not in model_groups:
            model_groups[model_params] = {
                'dataset_sizes': [],
                'memorization_values': [],
                'capacity_estimate': result['capacity_estimate']
            }
    
    # Plot each model size
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_groups)))
    
    for i, (model_params, data) in enumerate(sorted(model_groups.items())):
        capacity_est = data['capacity_estimate']
        dataset_sizes = capacity_est.dataset_sizes
        memorization_values = capacity_est.memorization_values
        
        # Plot memorization curve
        ax.plot(dataset_sizes, memorization_values, 
               'o-', color=colors[i], linewidth=2, markersize=6,
               label=f'{model_params:,} parameters')
        
        # Add plateau line
        plateau_value = capacity_est.estimated_capacity_bits
        ax.axhline(y=plateau_value, color=colors[i], linestyle='--', alpha=0.5)
        
        # Mark plateau detection point
        plateau_size = capacity_est.plateau_dataset_size
        ax.axvline(x=plateau_size, color=colors[i], linestyle=':', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Training Set Size (number of datapoints)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Unintended Memorization (bits)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
    
    # Add annotation
    ax.text(0.02, 0.98, 'Memorization plateaus at\nmodel capacity limits', 
           transform=ax.transAxes, fontsize=TICK_SIZE,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_capacity_vs_parameters(
    capacity_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Model Capacity vs Parameters (Morris et al. Figure 6 Reproduction)"
) -> plt.Figure:
    """
    Reproduce Morris et al. Figure 6: Model capacity vs parameters showing bits-per-parameter.
    
    Args:
        capacity_results: Results from run_capacity_experiments
        save_path: Optional path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure object
    """
    setup_figure_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    model_sizes = capacity_results['model_sizes']
    estimated_capacities = capacity_results['estimated_capacities']
    scaling_law = capacity_results['scaling_law']
    
    # Convert to numpy arrays
    model_sizes = np.array(model_sizes)
    estimated_capacities = np.array(estimated_capacities)
    
    # Plot observed data points
    ax.scatter(model_sizes, estimated_capacities, 
              s=100, color=MORRIS_COLORS['observed'], alpha=0.7, 
              label='Observed Capacity', zorder=5)
    
    # Plot fitted scaling law - minimal fix for empty arrays
    if len(model_sizes) > 0:
        x_fit = np.linspace(model_sizes.min(), model_sizes.max(), 100)
        y_fit = scaling_law['bits_per_parameter'] * x_fit + scaling_law['intercept']
        
        ax.plot(x_fit, y_fit, '--', color=MORRIS_COLORS['good_fit'], linewidth=2,
               label=f'Fitted Line: {scaling_law["bits_per_parameter"]:.2f} bits/param\n'
                     f'R² = {scaling_law["r_squared"]:.3f}')
        
        # Plot Morris et al. reference line
        y_morris = MORRIS_BITS_PER_PARAM * x_fit
        ax.plot(x_fit, y_morris, ':', color=MORRIS_COLORS['target'], linewidth=2,
               label=f'Morris et al.: {MORRIS_BITS_PER_PARAM} bits/param')
    
    # Add individual model annotations
    for i, (size, capacity) in enumerate(zip(model_sizes, estimated_capacities)):
        bpp = capacity / size
        ax.annotate(f'{bpp:.1f}', (size, capacity), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=TICK_SIZE-2, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Model Size (parameters)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Total Memorization (bits)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=LEGEND_SIZE)
    
    # Add summary statistics
    summary = capacity_results['summary_statistics']
    stats_text = f"Mean: {summary['mean_bits_per_parameter']:.2f} ± {summary['std_bits_per_parameter']:.2f} bits/param\n"
    stats_text += f"Target: {MORRIS_BITS_PER_PARAM} bits/param\n"
    stats_text += f"Deviation: {abs(summary['mean_bits_per_parameter'] - MORRIS_BITS_PER_PARAM):.2f}"
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=TICK_SIZE-1, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_scaling_law_validation(
    capacity_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create detailed scaling law validation plot with residuals and R² analysis.
    
    Args:
        capacity_results: Results from run_capacity_experiments
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    setup_figure_style()
    
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Extract data
    model_sizes = np.array(capacity_results['model_sizes'])
    estimated_capacities = np.array(capacity_results['estimated_capacities'])
    scaling_law = capacity_results['scaling_law']
    
    # Main scaling plot
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot data and fit
    ax1.scatter(model_sizes, estimated_capacities, s=80, color=MORRIS_COLORS['observed'], alpha=0.7)
    
    x_fit = np.linspace(model_sizes.min(), model_sizes.max(), 100)
    y_fit = scaling_law['bits_per_parameter'] * x_fit + scaling_law['intercept']
    ax1.plot(x_fit, y_fit, '--', color=MORRIS_COLORS['good_fit'], linewidth=2)
    
    # Morris reference
    y_morris = MORRIS_BITS_PER_PARAM * x_fit
    ax1.plot(x_fit, y_morris, ':', color=MORRIS_COLORS['target'], linewidth=2)
    
    ax1.set_xlabel('Model Parameters')
    ax1.set_ylabel('Capacity (bits)')
    ax1.set_title('Scaling Law Validation')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2 = fig.add_subplot(gs[1, :2])
    
    y_pred = scaling_law['bits_per_parameter'] * model_sizes + scaling_law['intercept']
    residuals = estimated_capacities - y_pred
    
    ax2.scatter(model_sizes, residuals, color=MORRIS_COLORS['observed'], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Model Parameters')
    ax2.set_ylabel('Residuals')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # R² and statistics panel
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis('off')
    
    # Calculate additional statistics
    r_squared = scaling_law['r_squared']
    slope = scaling_law['bits_per_parameter']
    intercept = scaling_law['intercept']
    
    mean_residual = np.mean(np.abs(residuals))
    max_residual = np.max(np.abs(residuals))
    
    stats_text = f"""Scaling Law Analysis
    
Fitted Relationship:
  Slope: {slope:.3f} bits/param
  Intercept: {intercept:.1f} bits
  R²: {r_squared:.4f}

Morris et al. Reference:
  Target: {MORRIS_BITS_PER_PARAM} bits/param
  Deviation: {abs(slope - MORRIS_BITS_PER_PARAM):.3f}
  Relative Error: {abs(slope - MORRIS_BITS_PER_PARAM)/MORRIS_BITS_PER_PARAM:.1%}

Residual Analysis:
  Mean |Residual|: {mean_residual:.1f}
  Max |Residual|: {max_residual:.1f}
  
Quality Assessment:
  {'✓ Excellent' if r_squared > 0.95 else '✓ Good' if r_squared > 0.8 else '⚠ Poor'} Fit (R² = {r_squared:.3f})
  {'✓ Close' if abs(slope - MORRIS_BITS_PER_PARAM) < 0.5 else '⚠ Distant'} to Morris Target
"""
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=TICK_SIZE-1,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=MORRIS_COLORS['background'], alpha=0.3))
    
    plt.suptitle('Scaling Law Validation: Capacity vs Model Size', fontsize=TITLE_SIZE)
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_training_dynamics(
    training_metrics: Dict[str, List[float]],
    memorization_values: List[float],
    dataset_sizes: List[int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training dynamics showing loss curves and memorization evolution.
    
    Args:
        training_metrics: Training metrics from model training
        memorization_values: Memorization values during training
        dataset_sizes: Dataset sizes tested
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    setup_figure_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss curves
    if 'train_loss' in training_metrics and len(training_metrics['train_loss']) > 0:
        steps = training_metrics.get('step', range(len(training_metrics['train_loss'])))
        losses = training_metrics['train_loss']
        
        ax1.plot(steps, losses, color=MORRIS_COLORS['observed'], linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'No training metrics available', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Training Loss Curve')
    
    # Memorization vs dataset size
    ax2.plot(dataset_sizes, memorization_values, 'o-', 
            color=MORRIS_COLORS['good_fit'], linewidth=2, markersize=8)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Total Memorization (bits)')
    ax2.set_title('Memorization vs Dataset Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add plateau detection
    if len(memorization_values) > 2:
        # Simple plateau detection for visualization
        plateau_value = max(memorization_values)
        plateau_threshold = 0.95 * plateau_value
        
        plateau_indices = [i for i, val in enumerate(memorization_values) if val >= plateau_threshold]
        if plateau_indices:
            plateau_start = dataset_sizes[plateau_indices[0]]
            ax2.axvline(x=plateau_start, color=MORRIS_COLORS['target'], 
                       linestyle='--', alpha=0.7, label=f'Plateau starts ~{plateau_start}')
            ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def create_experimental_dashboard(
    suite: ExperimentSuite,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive experimental dashboard with all key metrics.
    
    Args:
        suite: Complete experimental suite results
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    setup_figure_style()
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    capacity_results = suite.capacity_results
    validation = suite.validation_results
    comparison = suite.morris_comparison
    
    # 1. Main capacity vs parameters plot
    ax1 = fig.add_subplot(gs[0, :2])
    model_sizes = np.array(capacity_results['model_sizes'])
    estimated_capacities = np.array(capacity_results['estimated_capacities'])
    
    ax1.scatter(model_sizes, estimated_capacities, s=100, 
               color=MORRIS_COLORS['observed'], alpha=0.7)
    
    # Scaling law fit
    scaling_law = capacity_results['scaling_law']
    x_fit = np.linspace(model_sizes.min(), model_sizes.max(), 100)
    y_fit = scaling_law['bits_per_parameter'] * x_fit + scaling_law['intercept']
    ax1.plot(x_fit, y_fit, '--', color=MORRIS_COLORS['good_fit'], linewidth=2)
    
    ax1.set_xlabel('Model Parameters')
    ax1.set_ylabel('Capacity (bits)')
    ax1.set_title('Capacity vs Model Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Bits per parameter comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    observed_bpp = comparison['observed_bpp']
    target_bpp = comparison['morris_target_bpp']
    
    bars = ax2.bar(['Observed', 'Morris Target'], [observed_bpp, target_bpp],
                  color=[MORRIS_COLORS['observed'], MORRIS_COLORS['target']], alpha=0.7)
    
    ax2.set_ylabel('Bits per Parameter')
    ax2.set_title('Bits/Parameter Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, [observed_bpp, target_bpp]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}', ha='center', va='bottom')
    
    # 3. Validation status
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    
    validation_text = "Validation Results:\n\n"
    for key, value in validation.items():
        if key != 'experiment_valid':
            status = "✓" if value else "✗"
            clean_key = key.replace('_', ' ').title()
            validation_text += f"{status} {clean_key}\n"
    
    overall_status = "✓ PASSED" if validation['experiment_valid'] else "✗ FAILED"
    validation_text += f"\nOverall: {overall_status}"
    
    color = MORRIS_COLORS['good_fit'] if validation['experiment_valid'] else MORRIS_COLORS['poor_fit']
    ax3.text(0.05, 0.95, validation_text, transform=ax3.transAxes, fontsize=TICK_SIZE,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    # 4. Individual model memorization curves
    ax4 = fig.add_subplot(gs[1, :3])
    
    individual_results = capacity_results['individual_results']
    colors = plt.cm.viridis(np.linspace(0, 1, len(individual_results)))
    
    for i, result in enumerate(individual_results):
        capacity_est = result['capacity_estimate']
        ax4.plot(capacity_est.dataset_sizes, capacity_est.memorization_values,
                'o-', color=colors[i], label=f'{result["model_params"]:,} params')
    
    ax4.set_xlabel('Dataset Size')
    ax4.set_ylabel('Memorization (bits)')
    ax4.set_title('Memorization Curves by Model Size')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Reproduction score gauge
    ax5 = fig.add_subplot(gs[1, 3])
    
    score = comparison['morris_reproduction_score']
    
    # Create a simple gauge chart
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Color segments based on score ranges
    colors_gauge = []
    for angle in theta:
        score_at_angle = (angle / np.pi) * 100
        if score_at_angle < 50:
            colors_gauge.append(MORRIS_COLORS['poor_fit'])
        elif score_at_angle < 80:
            colors_gauge.append(MORRIS_COLORS['observed'])
        else:
            colors_gauge.append(MORRIS_COLORS['good_fit'])
    
    ax5.pie([score, 100-score], startangle=180, counterclock=False, 
           colors=[MORRIS_COLORS['good_fit'] if score > 80 else 
                  MORRIS_COLORS['observed'] if score > 50 else 
                  MORRIS_COLORS['poor_fit'], 'lightgray'],
           wedgeprops=dict(width=0.3))
    
    ax5.text(0, 0, f'{score:.1f}', ha='center', va='center', 
            fontsize=TITLE_SIZE, fontweight='bold')
    ax5.set_title('Reproduction Score\n(0-100)')
    
    # 6. Experiment configuration summary
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.axis('off')
    
    config = suite.experiment_config
    config_text = f"""Experiment Configuration:
    
Device: {config.device}
CPU Optimizations: {config.use_cpu_optimizations}
Max Model Size: {config.max_model_size:,} parameters
Max Dataset Size: {config.max_dataset_size:,} samples
Number of Seeds: {config.n_seeds}
Execution Time: {suite.execution_time:.1f} seconds

Morris et al. Comparison:
Target: {comparison['morris_target_bpp']} bits/param
Observed: {comparison['observed_bpp']:.2f} bits/param
Relative Error: {comparison['bpp_relative_error']:.1%}
Scaling R²: {capacity_results['scaling_law']['r_squared']:.3f}
"""
    
    ax6.text(0.05, 0.95, config_text, transform=ax6.transAxes, fontsize=TICK_SIZE-1,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=MORRIS_COLORS['background'], alpha=0.3))
    
    # 7. Summary statistics
    ax7 = fig.add_subplot(gs[2, 2:])
    
    summary = capacity_results['summary_statistics']
    
    # Create bar chart of key metrics
    metrics = ['Mean BPP', 'Std BPP', 'Scaling R²', 'Repro Score']
    values = [summary['mean_bits_per_parameter'], 
             summary['std_bits_per_parameter'],
             capacity_results['scaling_law']['r_squared'] * 4,  # Scale for visibility
             comparison['morris_reproduction_score'] / 25]  # Scale for visibility
    
    colors_bar = [MORRIS_COLORS['observed'], MORRIS_COLORS['good_fit'], 
                  MORRIS_COLORS['target'], MORRIS_COLORS['poor_fit']]
    
    bars = ax7.bar(metrics, values, color=colors_bar, alpha=0.7)
    ax7.set_ylabel('Normalized Values')
    ax7.set_title('Key Metrics Summary')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add actual values as text
    actual_values = [f'{summary["mean_bits_per_parameter"]:.2f}',
                    f'{summary["std_bits_per_parameter"]:.2f}',
                    f'{capacity_results["scaling_law"]["r_squared"]:.3f}',
                    f'{comparison["morris_reproduction_score"]:.1f}']
    
    for bar, actual in zip(bars, actual_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                actual, ha='center', va='bottom', fontsize=TICK_SIZE-2)
    
    # Main title
    plt.suptitle(f'Morris et al. Reproduction Dashboard - {suite.suite_name}', 
                fontsize=TITLE_SIZE+2, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
    
    return fig


def compare_to_morris_results(
    our_results: Dict[str, Any],
    morris_reference: Dict[str, float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create side-by-side comparison with Morris et al. published results.
    
    Args:
        our_results: Our experimental results
        morris_reference: Morris et al. reference values
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    if morris_reference is None:
        morris_reference = {
            'bits_per_parameter': MORRIS_BITS_PER_PARAM,
            'scaling_r_squared': 0.95,
            'model_sizes': [170000, 500000, 2500000, 7000000],
            'capacities': [612000, 1800000, 9000000, 25200000]
        }
    
    setup_figure_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bits per parameter comparison
    our_bpp = our_results['summary_statistics']['mean_bits_per_parameter']
    our_std = our_results['summary_statistics']['std_bits_per_parameter']
    morris_bpp = morris_reference['bits_per_parameter']
    
    x = ['Our Results', 'Morris et al.']
    y = [our_bpp, morris_bpp]
    yerr = [our_std, 0]
    
    bars = ax1.bar(x, y, yerr=yerr, capsize=5,
                  color=[MORRIS_COLORS['observed'], MORRIS_COLORS['target']], alpha=0.7)
    ax1.set_ylabel('Bits per Parameter')
    ax1.set_title('Bits per Parameter Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value, error in zip(bars, y, yerr):
        height = bar.get_height()
        if error > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + error + 0.05,
                    f'{value:.2f}±{error:.2f}', ha='center', va='bottom')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # 2. Scaling law quality comparison
    our_r2 = our_results['scaling_law']['r_squared']
    morris_r2 = morris_reference['scaling_r_squared']
    
    bars = ax2.bar(['Our R²', 'Morris R²'], [our_r2, morris_r2],
                  color=[MORRIS_COLORS['observed'], MORRIS_COLORS['target']], alpha=0.7)
    ax2.set_ylabel('R² Value')
    ax2.set_title('Scaling Law Quality')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, [our_r2, morris_r2]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Capacity vs parameters - our results
    our_sizes = np.array(our_results['model_sizes'])
    our_capacities = np.array(our_results['estimated_capacities'])
    
    ax3.scatter(our_sizes, our_capacities, s=80, color=MORRIS_COLORS['observed'], alpha=0.7)
    
    # Our scaling law
    x_fit = np.linspace(our_sizes.min(), our_sizes.max(), 100)
    y_fit = our_results['scaling_law']['bits_per_parameter'] * x_fit + our_results['scaling_law']['intercept']
    ax3.plot(x_fit, y_fit, '--', color=MORRIS_COLORS['good_fit'], linewidth=2)
    
    ax3.set_xlabel('Model Parameters')
    ax3.set_ylabel('Capacity (bits)')
    ax3.set_title('Our Results: Capacity vs Parameters')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Morris et al. reference (reconstructed)
    morris_sizes = np.array(morris_reference['model_sizes'])
    morris_capacities = np.array(morris_reference['capacities'])
    
    ax4.scatter(morris_sizes, morris_capacities, s=80, color=MORRIS_COLORS['target'], alpha=0.7)
    
    # Morris scaling law
    x_morris = np.linspace(morris_sizes.min(), morris_sizes.max(), 100)
    y_morris = morris_reference['bits_per_parameter'] * x_morris
    ax4.plot(x_morris, y_morris, '--', color=MORRIS_COLORS['good_fit'], linewidth=2)
    
    ax4.set_xlabel('Model Parameters')
    ax4.set_ylabel('Capacity (bits)')
    ax4.set_title('Morris et al.: Capacity vs Parameters')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison with Morris et al. Results', fontsize=TITLE_SIZE+2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    return fig


def save_all_figures(
    suite: ExperimentSuite,
    output_dir: str = "figures",
    prefix: str = "morris_reproduction"
) -> List[str]:
    """
    Generate and save all visualization figures for an experimental suite.
    
    Args:
        suite: Complete experimental suite
        output_dir: Directory to save figures
        prefix: Prefix for figure filenames
        
    Returns:
        List of saved figure paths
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_paths = []
    
    # Figure 1: Memorization vs dataset size
    fig1_path = os.path.join(output_dir, f"{prefix}_figure1_{timestamp}.png")
    plot_memorization_vs_dataset_size(suite.capacity_results, save_path=fig1_path)
    saved_paths.append(fig1_path)
    plt.close()
    
    # Figure 6: Capacity vs parameters
    fig6_path = os.path.join(output_dir, f"{prefix}_figure6_{timestamp}.png")
    plot_capacity_vs_parameters(suite.capacity_results, save_path=fig6_path)
    saved_paths.append(fig6_path)
    plt.close()
    
    # Scaling law validation
    scaling_path = os.path.join(output_dir, f"{prefix}_scaling_laws_{timestamp}.png")
    plot_scaling_law_validation(suite.capacity_results, save_path=scaling_path)
    saved_paths.append(scaling_path)
    plt.close()
    
    # Experimental dashboard
    dashboard_path = os.path.join(output_dir, f"{prefix}_dashboard_{timestamp}.png")
    create_experimental_dashboard(suite, save_path=dashboard_path)
    saved_paths.append(dashboard_path)
    plt.close()
    
    # Morris comparison
    comparison_path = os.path.join(output_dir, f"{prefix}_morris_comparison_{timestamp}.png")
    compare_to_morris_results(suite.capacity_results, save_path=comparison_path)
    saved_paths.append(comparison_path)
    plt.close()
    
    print(f"\nAll figures saved to: {output_dir}")
    print(f"Generated {len(saved_paths)} figures:")
    for path in saved_paths:
        print(f"  - {os.path.basename(path)}")
    
    return saved_paths


def create_publication_figure_set(
    suite: ExperimentSuite,
    output_dir: str = "publication_figures"
) -> Dict[str, str]:
    """
    Create publication-quality figure set ready for academic papers.
    
    Args:
        suite: Experimental suite results
        output_dir: Output directory for figures
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # High-resolution settings for publication
    original_dpi = plt.rcParams['figure.dpi']
    plt.rcParams['figure.dpi'] = 600
    
    figures = {}
    
    try:
        # Main result figures matching Morris et al.
        fig1_path = os.path.join(output_dir, "figure_1_memorization_plateaus.pdf")
        plot_memorization_vs_dataset_size(
            suite.capacity_results, 
            save_path=fig1_path,
            title="Unintended Memorization of Uniform Random Data"
        )
        figures["Figure 1"] = fig1_path
        plt.close()
        
        fig6_path = os.path.join(output_dir, "figure_6_capacity_scaling.pdf")
        plot_capacity_vs_parameters(
            suite.capacity_results,
            save_path=fig6_path,
            title="Capacity in Bits-per-Parameter"
        )
        figures["Figure 6"] = fig6_path
        plt.close()
        
        # Validation figure
        validation_path = os.path.join(output_dir, "scaling_law_validation.pdf")
        plot_scaling_law_validation(suite.capacity_results, save_path=validation_path)
        figures["Scaling Validation"] = validation_path
        plt.close()
        
        print(f"Publication figures saved to: {output_dir}")
        
    finally:
        # Restore original DPI
        plt.rcParams['figure.dpi'] = original_dpi
    
    return figures


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded. Use with experimental results to create figures.")
    print("\nExample usage:")
    print("from experiment_runner import run_morris_reproduction_suite")
    print("from visualization import save_all_figures")
    print()
    print("suite = run_morris_reproduction_suite()")
    print("save_all_figures(suite)")
