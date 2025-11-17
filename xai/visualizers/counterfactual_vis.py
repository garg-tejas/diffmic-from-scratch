"""
Counterfactual Visualizer Module

This module creates visualizations for counterfactual explanations,
showing the original image, counterfactual, and delta map.

Purpose:
- Visualize counterfactual comparisons
- Show delta maps highlighting changes
- Create transition matrices for grade changes
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CounterfactualVisualizer:
    """
    Visualize counterfactual explanations.
    
    Attributes:
        config: Visualization configuration dictionary
        
    Usage:
        >>> vis = CounterfactualVisualizer(config)
        >>> vis.visualize_counterfactual(original, counterfactual, delta_map, 'output.png')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize counterfactual visualizer.
        
        Args:
            config: Configuration with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'jet')
        self.figure_size = config.get('figure_size', [12, 8])
        self.dpi = config.get('image_dpi', 300)
        self.class_names = config.get('class_names', {})
    
    def visualize_counterfactual(self,
                                original: np.ndarray,
                                counterfactual: np.ndarray,
                                delta_map: np.ndarray,
                                save_path: Path,
                                original_class: Optional[int] = None,
                                counterfactual_class: Optional[int] = None) -> None:
        """
        Visualize counterfactual comparison.
        
        Args:
            original: Original image (H, W, C) or (C, H, W)
            counterfactual: Counterfactual image
            delta_map: Difference map (H, W)
            save_path: Path to save figure
            original_class: Original class label
            counterfactual_class: Counterfactual class label
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Normalize images
        if original.ndim == 3 and original.shape[0] == 3:
            original = original.transpose(1, 2, 0)
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        
        # Original image
        axes[0].imshow(original)
        orig_title = 'Original'
        if original_class is not None:
            class_name = self.class_names.get(str(original_class), f"Class {original_class}")
            orig_title += f' ({class_name})'
        axes[0].set_title(orig_title, fontsize=12)
        axes[0].axis('off')
        
        # Counterfactual (if it's a probability map, visualize as heatmap)
        if counterfactual.ndim == 3:
            # Average over classes or show max class
            counterfactual_vis = counterfactual.max(axis=0) if counterfactual.shape[0] > 1 else counterfactual[0]
            im = axes[1].imshow(counterfactual_vis, cmap=self.colormap)
            plt.colorbar(im, ax=axes[1])
        else:
            axes[1].imshow(counterfactual, cmap='gray')
        
        cf_title = 'Counterfactual'
        if counterfactual_class is not None:
            class_name = self.class_names.get(str(counterfactual_class), f"Class {counterfactual_class}")
            cf_title += f' ({class_name})'
        axes[1].set_title(cf_title, fontsize=12)
        axes[1].axis('off')
        
        # Delta map
        im = axes[2].imshow(delta_map, cmap='hot')
        axes[2].set_title('Delta Map (Changes)', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], label='Difference Magnitude')
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_grade_transition_matrix(self,
                                      all_counterfactuals: List[Dict[str, Any]],
                                      save_path: Path) -> None:
        """
        Create matrix showing transitions between grades.
        
        Args:
            all_counterfactuals: List of counterfactual results
            save_path: Path to save figure
        """
        # Count transitions
        num_classes = 5  # APTOS has 5 classes
        transition_matrix = np.zeros((num_classes, num_classes))
        
        for cf in all_counterfactuals:
            orig = cf.get('original_prediction', -1)
            target = cf.get('target_class', -1)
            if 0 <= orig < num_classes and 0 <= target < num_classes:
                transition_matrix[orig, target] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix_norm = transition_matrix / row_sums
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Raw counts
        im1 = ax1.imshow(transition_matrix, cmap='Blues')
        ax1.set_xlabel('Target Class', fontsize=12)
        ax1.set_ylabel('Original Class', fontsize=12)
        ax1.set_title('Transition Counts', fontsize=14)
        plt.colorbar(im1, ax=ax1)
        
        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                if transition_matrix[i, j] > 0:
                    ax1.text(j, i, int(transition_matrix[i, j]),
                           ha='center', va='center', color='white', fontsize=10)
        
        # Normalized probabilities
        im2 = ax2.imshow(transition_matrix_norm, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xlabel('Target Class', fontsize=12)
        ax2.set_ylabel('Original Class', fontsize=12)
        ax2.set_title('Transition Probabilities', fontsize=14)
        plt.colorbar(im2, ax=ax2)
        
        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                if transition_matrix_norm[i, j] > 0:
                    ax2.text(j, i, f'{transition_matrix_norm[i, j]:.2f}',
                           ha='center', va='center', color='white', fontsize=10)
        
        # Set tick labels
        class_labels = [self.class_names.get(str(i), f"G{i}") for i in range(num_classes)]
        for ax in [ax1, ax2]:
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels(class_labels)
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels(class_labels)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_counterfactual_trajectory(self,
                                       trajectory: List[Dict],
                                       save_path: Path) -> None:
        """
        Plot counterfactual generation trajectory.
        
        Args:
            trajectory: Denoising trajectory showing prediction changes
            save_path: Path to save figure
        """
        if not trajectory:
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        timesteps = [step['timestep'] for step in trajectory]
        predictions = [step['prediction'] for step in trajectory]
        confidences = [step['confidence'] for step in trajectory]
        
        ax.plot(timesteps, predictions, 'b-', linewidth=2, marker='o', label='Prediction')
        ax2 = ax.twinx()
        ax2.plot(timesteps, confidences, 'r--', linewidth=2, marker='s', label='Confidence')
        
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Predicted Class', fontsize=12, color='b')
        ax2.set_ylabel('Confidence', fontsize=12, color='r')
        ax.set_title('Counterfactual Generation Trajectory', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

