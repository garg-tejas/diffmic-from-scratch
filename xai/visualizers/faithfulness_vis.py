"""
Faithfulness Visualizer Module

This module creates visualizations for faithfulness validation results,
including insertion/deletion curves and robustness comparisons.

Purpose:
- Plot AUC curves for deletion and insertion games
- Visualize occlusion sequences
- Compare robustness across different augmentations
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FaithfulnessVisualizer:
    """
    Visualize faithfulness validation results.
    
    Attributes:
        config: Visualization configuration dictionary
        
    Usage:
        >>> vis = FaithfulnessVisualizer(config)
        >>> vis.plot_insertion_deletion_curves(deletion_curve, insertion_curve, 'output.png')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize faithfulness visualizer.
        
        Args:
            config: Configuration with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'jet')
        self.figure_size = config.get('figure_size', [12, 8])
        self.dpi = config.get('image_dpi', 300)
    
    def plot_insertion_deletion_curves(self,
                                       deletion_curve: List[Tuple[float, float]],
                                       insertion_curve: List[Tuple[float, float]],
                                       deletion_auc: float,
                                       insertion_auc: float,
                                       save_path: Path) -> None:
        """
        Plot insertion and deletion curves on the same figure.
        
        Args:
            deletion_curve: List of (occlusion_pct, confidence) tuples
            insertion_curve: List of (reveal_pct, confidence) tuples
            deletion_auc: Area under deletion curve
            insertion_auc: Area under insertion curve
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Deletion curve
        del_pcts = [p[0] for p in deletion_curve]
        del_confs = [p[1] for p in deletion_curve]
        
        ax1.plot(del_pcts, del_confs, 'r-', linewidth=2, label='Deletion')
        ax1.fill_between(del_pcts, del_confs, alpha=0.3, color='red')
        ax1.set_xlabel('Occlusion Percentage (%)', fontsize=12)
        ax1.set_ylabel('Confidence', fontsize=12)
        ax1.set_title(f'Deletion Game (AUC: {deletion_auc:.3f})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, max(del_confs) * 1.1])
        
        # Insertion curve
        ins_pcts = [p[0] for p in insertion_curve]
        ins_confs = [p[1] for p in insertion_curve]
        
        ax2.plot(ins_pcts, ins_confs, 'g-', linewidth=2, label='Insertion')
        ax2.fill_between(ins_pcts, ins_confs, alpha=0.3, color='green')
        ax2.set_xlabel('Reveal Percentage (%)', fontsize=12)
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_title(f'Insertion Game (AUC: {insertion_auc:.3f})', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, max(ins_confs) * 1.1])
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_robustness_comparison(self,
                                   original_saliency: np.ndarray,
                                   augmented_saliencies: Dict[str, np.ndarray],
                                   save_path: Path) -> None:
        """
        Plot comparison of saliency maps under different augmentations.
        
        Args:
            original_saliency: Original saliency map (H, W)
            augmented_saliencies: Dictionary of {aug_name: saliency_map}
            save_path: Path to save figure
        """
        n_augs = len(augmented_saliencies)
        fig, axes = plt.subplots(1, n_augs + 1, figsize=(4 * (n_augs + 1), 4))
        
        if n_augs == 0:
            axes = [axes]
        
        # Original
        axes[0].imshow(original_saliency, cmap=self.colormap)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')
        
        # Augmented
        for idx, (aug_name, aug_saliency) in enumerate(augmented_saliencies.items(), 1):
            axes[idx].imshow(aug_saliency, cmap=self.colormap)
            axes[idx].set_title(aug_name, fontsize=12)
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_occlusion_animation(self,
                                   image: np.ndarray,
                                   saliency_map: np.ndarray,
                                   occlusion_sequence: List[np.ndarray],
                                   save_path: Path,
                                   fps: int = 2) -> None:
        """
        Create animation showing progressive occlusion.
        
        Args:
            image: Original image (H, W, 3)
            saliency_map: Saliency map (H, W)
            occlusion_sequence: List of occluded images at each step
            save_path: Path to save GIF
            fps: Frames per second
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        def animate(frame):
            axes[0].clear()
            axes[1].clear()
            
            # Show saliency
            axes[0].imshow(saliency_map, cmap=self.colormap)
            axes[0].set_title('Saliency Map', fontsize=12)
            axes[0].axis('off')
            
            # Show occluded image
            if frame < len(occlusion_sequence):
                occluded = occlusion_sequence[frame]
                axes[1].imshow(occluded)
                axes[1].set_title(f'Occlusion Step {frame + 1}/{len(occlusion_sequence)}', fontsize=12)
                axes[1].axis('off')
            else:
                axes[1].imshow(image)
                axes[1].set_title('Original', fontsize=12)
                axes[1].axis('off')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(occlusion_sequence) + 1, interval=1000/fps, repeat=True)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
    
    def plot_faithfulness_summary(self,
                                  faithfulness_results: Dict[str, Any],
                                  save_path: Path) -> None:
        """
        Create summary plot of all faithfulness metrics.
        
        Args:
            faithfulness_results: Dictionary with deletion_auc, insertion_auc, robustness_scores
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Deletion AUC (lower is better)
        del_auc = faithfulness_results.get('deletion_auc', 0.0)
        axes[0, 0].barh(['Deletion AUC'], [del_auc], color='red', alpha=0.7)
        axes[0, 0].set_xlabel('AUC Score', fontsize=12)
        axes[0, 0].set_title('Deletion Game (Lower = Better)', fontsize=14)
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].axvline(x=0.1, color='green', linestyle='--', label='Excellent (<0.1)')
        axes[0, 0].axvline(x=0.3, color='yellow', linestyle='--', label='Good (<0.3)')
        axes[0, 0].legend()
        
        # Insertion AUC (higher is better)
        ins_auc = faithfulness_results.get('insertion_auc', 0.0)
        axes[0, 1].barh(['Insertion AUC'], [ins_auc], color='green', alpha=0.7)
        axes[0, 1].set_xlabel('AUC Score', fontsize=12)
        axes[0, 1].set_title('Insertion Game (Higher = Better)', fontsize=14)
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].axvline(x=0.9, color='green', linestyle='--', label='Excellent (>0.9)')
        axes[0, 1].axvline(x=0.7, color='yellow', linestyle='--', label='Good (>0.7)')
        axes[0, 1].legend()
        
        # Robustness scores
        robustness = faithfulness_results.get('robustness_scores', {})
        if robustness:
            aug_names = []
            corr_scores = []
            for key in ['brightness_correlation', 'contrast_correlation', 'rotation_correlation']:
                if key in robustness:
                    aug_names.append(key.replace('_correlation', '').title())
                    corr_scores.append(robustness[key])
            
            if aug_names:
                axes[1, 0].bar(aug_names, corr_scores, color='blue', alpha=0.7)
                axes[1, 0].set_ylabel('Pearson Correlation', fontsize=12)
                axes[1, 0].set_title('Semantic Robustness', fontsize=14)
                axes[1, 0].set_ylim([0, 1])
                axes[1, 0].axhline(y=0.9, color='green', linestyle='--', label='Excellent (>0.9)')
                axes[1, 0].axhline(y=0.7, color='yellow', linestyle='--', label='Good (>0.7)')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall trust score
        overall_score = faithfulness_results.get('overall_trust_score', 0.0)
        axes[1, 1].barh(['Overall Trust'], [overall_score], color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Trust Score', fontsize=12)
        axes[1, 1].set_title('Overall Trust Score', fontsize=14)
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].axvline(x=0.8, color='green', linestyle='--', label='High (>0.8)')
        axes[1, 1].axvline(x=0.6, color='yellow', linestyle='--', label='Medium (>0.6)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
