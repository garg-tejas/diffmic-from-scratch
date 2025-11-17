"""
Attribution Visualizer Module

This module creates visualizations for conditional attribution results,
showing global vs local contributions and guidance map attributions.

Purpose:
- Plot feature attribution bars (global vs ROI contributions)
- Visualize guidance map attribution heatmaps
- Create comparison plots for different samples
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AttributionVisualizer:
    """
    Visualize conditional attribution results.
    
    Attributes:
        config: Visualization configuration dictionary
        
    Usage:
        >>> vis = AttributionVisualizer(config)
        >>> vis.plot_feature_attribution_bars(raw_score, roi_scores, 'output.png')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize attribution visualizer.
        
        Args:
            config: Configuration with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'jet')
        self.figure_size = config.get('figure_size', [12, 8])
        self.dpi = config.get('image_dpi', 300)
        self.class_names = config.get('class_names', {})
    
    def plot_feature_attribution_bars(self,
                                     raw_score: float,
                                     roi_scores: List[float],
                                     save_path: Path,
                                     title: Optional[str] = None) -> None:
        """
        Plot bar chart showing feature attribution scores.
        
        Args:
            raw_score: Attribution to global (F_raw) prior
            roi_scores: List of attribution scores for each ROI
            save_path: Path to save figure
            title: Optional title for the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Prepare data
        labels = ['Global (F_raw)'] + [f'ROI {i+1}' for i in range(len(roi_scores))]
        scores = [raw_score] + roi_scores
        colors = ['blue'] + ['green'] * len(roi_scores)
        
        # Create bars
        bars = ax.bar(labels, scores, color=colors, alpha=0.7)
        
        # Highlight dominant ROI
        if len(roi_scores) > 0:
            dominant_idx = np.argmax(roi_scores) + 1  # +1 for global
            bars[dominant_idx].set_color('red')
            bars[dominant_idx].set_alpha(1.0)
        
        # Formatting
        ax.set_ylabel('Attribution Score', fontsize=12)
        ax.set_xlabel('Feature Component', fontsize=12)
        ax.set_title(title or 'Feature Attribution Scores', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(scores) * 1.1])
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_guidance_attribution_heatmap(self,
                                         M_attribution: np.ndarray,
                                         save_path: Path,
                                         highlight_diagonal: bool = True,
                                         title: Optional[str] = None) -> None:
        """
        Plot heatmap of guidance map attribution.
        
        Args:
            M_attribution: Attribution map (np, np) for guidance map M
            save_path: Path to save figure
            highlight_diagonal: Whether to highlight diagonal (global-dominant)
            title: Optional title for the plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot heatmap
        im = ax.imshow(M_attribution, cmap=self.colormap, aspect='auto')
        
        # Highlight diagonal if requested
        if highlight_diagonal:
            np_patches = M_attribution.shape[0]
            for i in range(np_patches):
                ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, 
                                         fill=False, edgecolor='white', linewidth=2))
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attribution Score')
        
        # Formatting
        ax.set_xlabel('Spatial Position (j)', fontsize=12)
        ax.set_ylabel('Spatial Position (i)', fontsize=12)
        ax.set_title(title or 'Guidance Map Attribution', fontsize=14)
        
        # Add text annotations for high values
        threshold = M_attribution.max() * 0.5
        for i in range(M_attribution.shape[0]):
            for j in range(M_attribution.shape[1]):
                if M_attribution[i, j] > threshold:
                    ax.text(j, i, f'{M_attribution[i, j]:.2f}',
                           ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_attribution_comparison(self,
                                     ground_truth_class: int,
                                     prediction: int,
                                     attributions: Dict[str, Any],
                                     save_path: Path) -> None:
        """
        Create comprehensive comparison plot of attribution results.
        
        Args:
            ground_truth_class: Ground truth class
            prediction: Predicted class
            attributions: Attribution results dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Feature attribution bars
        raw_score = attributions.get('global_contribution', 0.0)
        roi_scores = attributions.get('roi_contribution_scores', [])
        
        labels = ['Global'] + [f'ROI {i+1}' for i in range(len(roi_scores))]
        scores = [raw_score] + roi_scores
        colors = ['blue'] + ['green'] * len(roi_scores)
        
        if len(roi_scores) > 0:
            dominant_idx = np.argmax(roi_scores) + 1
            colors[dominant_idx] = 'red'
        
        axes[0, 0].bar(labels, scores, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Attribution Score', fontsize=12)
        axes[0, 0].set_title('Feature Attribution', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Global vs Local pie chart
        global_contrib = attributions.get('global_contribution', 0.0)
        local_contrib = attributions.get('local_contribution', 0.0)
        
        axes[0, 1].pie([global_contrib, local_contrib],
                       labels=['Global', 'Local'],
                       autopct='%1.1f%%',
                       startangle=90,
                       colors=['blue', 'green'])
        axes[0, 1].set_title('Global vs Local Contribution', fontsize=14)
        
        # 3. Guidance map attribution heatmap
        M_attribution = attributions.get('guidance_map_attribution')
        if M_attribution is not None:
            im = axes[1, 0].imshow(M_attribution, cmap=self.colormap, aspect='auto')
            axes[1, 0].set_title('Guidance Map Attribution', fontsize=14)
            axes[1, 0].set_xlabel('Spatial Position (j)', fontsize=12)
            axes[1, 0].set_ylabel('Spatial Position (i)', fontsize=12)
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Summary text
        axes[1, 1].axis('off')
        
        summary_text = f"""
        Attribution Summary
        
        Prediction: {prediction}
        Ground Truth: {ground_truth_class}
        
        Global Contribution: {global_contrib:.1%}
        Local Contribution: {local_contrib:.1%}
        
        Dominant ROI: {attributions.get('dominant_roi', -1) + 1}
        Guidance Strategy: {attributions.get('guidance_strategy', 'unknown')}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                       verticalalignment='center',
                       family='monospace')
        
        # Overall title
        gt_name = self.class_names.get(str(ground_truth_class), f"Class {ground_truth_class}")
        pred_name = self.class_names.get(str(prediction), f"Class {prediction}")
        fig.suptitle(f'Attribution Analysis: GT={gt_name}, Pred={pred_name}', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

