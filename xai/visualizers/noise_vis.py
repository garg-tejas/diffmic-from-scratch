"""
Noise Visualization Module

Creates visualizations for heterologous noise patterns and noise reduction
during the diffusion denoising process.

Features:
- Noise interaction maps showing spatial correlations
- Timestep distribution heatmaps
- Noise magnitude evolution plots
- Spatial noise pattern visualizations
- Noise reduction animations through diffusion timesteps
- Multi-panel noise analysis views

Purpose:
- Visualize how heterologous noise affects diffusion
- Understand noise reduction process
- Analyze spatial noise patterns and interactions
- Support research on diffusion dynamics
- Enable noise pattern debugging and analysis

Future Extensions:
- 3D noise visualization
- Interactive noise exploration
- Noise pattern clustering visualization
- Multi-sample noise comparison
- Noise-guided feature overlays
- Export to video formats (MP4, WebM)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, List
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.utils.image_utils import figure_to_array
from xai.utils.animation_utils import capture_frame
from xai.utils.pdf_utils import (
    save_figure_as_pdf,
    save_animation_frames_as_pdf,
    save_frame_grid_as_image,
    select_key_frames,
    apply_publication_style
)


class NoiseVisualizer:
    """
    Visualize heterologous noise patterns and noise reduction.
    
    Creates visualizations showing:
    - Noise interaction maps
    - Timestep distributions
    - Noise magnitude evolution
    - Spatial noise patterns
    - Noise reduction animations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the noise visualizer.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'viridis')
        self.figsize = config.get('figure_size', [12, 8])
        self.class_names = config.get('class_names', {})
        self.save_pdf = config.get('save_pdf', True)
        self.dpi = config.get('image_dpi', 300)
        
        # Apply publication style
        apply_publication_style()
        
        if not self.class_names:
            self.class_names = {str(i): f"Class {i}" for i in range(10)}
    
    def create_noise_interaction_map(self,
                                    noise_map: np.ndarray,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Show how noise points interact via spatial correlation.
        
        Args:
            noise_map: Noise map [C, H, W] or [H, W]
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        if noise_map.ndim == 3:
            # Average over classes or show for first class
            noise_map = noise_map[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Noise map
        im1 = ax1.imshow(noise_map, cmap='hot', aspect='auto')
        ax1.set_title('Noise Magnitude Map', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Width', fontsize=12)
        ax1.set_ylabel('Height', fontsize=12)
        plt.colorbar(im1, ax=ax1, label='Noise Magnitude')
        
        # Plot 2: Correlation with neighbors
        h, w = noise_map.shape
        flat_noise = noise_map.flatten()
        
        # Compute correlation with shifted versions
        shifts = [1, w, w+1, w-1]  # Right, down, diagonal
        correlations = []
        for shift in shifts:
            shifted = np.roll(flat_noise, shift)
            if shift > 0:
                shifted[:shift] = 0
            elif shift < 0:
                shifted[shift:] = 0
            corr = np.corrcoef(flat_noise, shifted)[0, 1]
            correlations.append(corr)
        
        # Create correlation visualization
        directions = ['Right', 'Down', 'Diag+', 'Diag-']
        bars = ax2.bar(directions, correlations, color='steelblue', alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Correlation', fontsize=12)
        ax2.set_title('Spatial Noise Correlation', fontsize=14, fontweight='bold')
        ax2.set_ylim(-1, 1)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.05,
                    f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.3)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_timestep_distribution(self,
                                    timestep_map: np.ndarray,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize random timestep sampling across spatial locations.
        
        Args:
            timestep_map: Timestep values at each location [H, W]
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(timestep_map, cmap='plasma', aspect='auto')
        ax.set_title('Timestep Distribution Across Spatial Locations', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width', fontsize=12)
        ax.set_ylabel('Height', fontsize=12)
        plt.colorbar(im, ax=ax, label='Timestep')
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_noise_magnitude_plot(self,
                                   noise_magnitudes: List[float],
                                   timesteps: List[int],
                                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Show noise magnitude evolution during denoising.
        
        Args:
            noise_magnitudes: List of noise magnitudes at each timestep
            timesteps: List of timestep values
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Reverse timesteps for display (from high to low)
        if len(timesteps) > 1 and timesteps[0] > timesteps[-1]:
            # Already in descending order
            x_vals = timesteps
        else:
            # Reverse if needed
            x_vals = timesteps[::-1]
            noise_magnitudes = noise_magnitudes[::-1]
        
        ax.plot(x_vals, noise_magnitudes, marker='o', linewidth=2, markersize=4, color='coral')
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Noise Magnitude', fontsize=12)
        ax.set_title('Noise Reduction Through Denoising', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(x_vals[0], x_vals[-1])
        
        # Add trend line
        if len(noise_magnitudes) > 1:
            z = np.polyfit(x_vals, noise_magnitudes, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), "r--", alpha=0.5, label=f'Trend (slope: {z[0]:.4f})')
            ax.legend(fontsize=10)
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_spatial_noise_animation(self,
                                      noise_data: List[Dict[str, Any]],
                                      save_path: Optional[str] = None,
                                      fps: int = 1) -> str:
        """
        Animate noise reduction across 2D space through diffusion timesteps.
        
        Args:
            noise_data: List of noise data dictionaries from NoiseExplainer
            save_path: Path to save GIF
            fps: Frames per second (default 1 for slow viewing)
            
        Returns:
            Path to saved GIF
        """
        import matplotlib
        matplotlib.use('Agg')
        
        if len(noise_data) == 0:
            raise ValueError("No noise data provided for animation")
        
        frames = []
        num_frames = len(noise_data)
        
        for frame_idx, step in enumerate(noise_data):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12), dpi=100)
            
            timestep = step.get('timestep', frame_idx)
            noise_map = step.get('noise_map', None)
            noise_magnitude = step.get('noise_magnitude', None)
            timestep_map = step.get('timestep_map', None)
            probs = step.get('probs', None)
            pred_class = step.get('predicted_class', 0)
            
            # Top left: Noise magnitude map
            if noise_magnitude is not None:
                im1 = ax1.imshow(noise_magnitude, cmap='hot', aspect='auto')
                ax1.set_title('Noise Magnitude', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Width', fontsize=9)
                ax1.set_ylabel('Height', fontsize=9)
                plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # Top right: Timestep map
            if timestep_map is not None:
                im2 = ax2.imshow(timestep_map, cmap='plasma', aspect='auto')
                ax2.set_title('Timestep Distribution', fontsize=11, fontweight='bold')
                ax2.set_xlabel('Width', fontsize=9)
                ax2.set_ylabel('Height', fontsize=9)
                plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # Bottom left: Noise map (per class if available)
            if noise_map is not None:
                if noise_map.ndim == 3:
                    # Show for predicted class
                    noise_class = noise_map[pred_class]
                else:
                    noise_class = noise_map
                
                im3 = ax3.imshow(noise_class, cmap='viridis', aspect='auto')
                ax3.set_title(f'Noise Map (Class {pred_class})', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Width', fontsize=9)
                ax3.set_ylabel('Height', fontsize=9)
                plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # Bottom right: Class probabilities
            if probs is not None:
                num_classes = len(probs)
                colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
                bar_colors = [colors[i] if i != pred_class else 'lightcoral' for i in range(num_classes)]
                
                bars = ax4.bar(range(num_classes), probs, color=bar_colors, edgecolor='black', linewidth=1)
                bars[pred_class].set_edgecolor('red')
                bars[pred_class].set_linewidth(3)
                
                ax4.set_xticks(range(num_classes))
                ax4.set_xticklabels([self.class_names.get(str(i), f"C{i}") for i in range(num_classes)],
                                  rotation=45, ha='right', fontsize=8)
                ax4.set_ylabel('Probability', fontsize=10)
                ax4.set_title('Class Probabilities', fontsize=11, fontweight='bold')
                ax4.set_ylim(0, 1)
                ax4.grid(axis='y', alpha=0.3)
            
            fig.suptitle(f'Noise Evolution - Timestep: {timestep} | Step: {frame_idx+1}/{num_frames}',
                        fontsize=14, fontweight='bold')
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08, hspace=0.3, wspace=0.3)
            
            frame = capture_frame(fig)
            frames.append(frame)
            plt.close(fig)
            
            if (frame_idx + 1) % max(1, num_frames // 10) == 0:
                print(f"  Frame {frame_idx + 1}/{num_frames} captured")
        
        if len(frames) == 0:
            raise ValueError("No frames generated for noise evolution visualization")
        
        layout = (5, 2)
        max_panels = layout[0] * layout[1]
        frame_indices = select_key_frames(list(range(len(frames))), max_panels)
        frame_indices = sorted(frame_indices)
        selected_frames = [frames[i] for i in frame_indices]
        
        frame_titles = []
        for idx in frame_indices:
            step = noise_data[idx]
            timestep = step.get('timestep', idx)
            frame_titles.append(f"t={timestep}")
        
        if save_path is None:
            save_path = 'noise_evolution.png'
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        save_frame_grid_as_image(
            selected_frames,
            save_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Noise Evolution Through Diffusion Timesteps',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[NoiseVisualizer] Stacked noise frames saved to {save_path}")
        
        if self.save_pdf:
            pdf_path = save_path.with_suffix('.pdf')
            save_animation_frames_as_pdf(
                selected_frames,
                pdf_path,
                layout=layout,
                titles=frame_titles,
                suptitle='Noise Evolution Through Diffusion Timesteps',
                dpi=self.dpi,
                frame_size=(3.5, 3.5),
                spacing=0.4
            )
            print(f"[NoiseVisualizer] PDF with frames saved to {pdf_path}")
        
        return save_path
    
    def create_noise_sensitivity_comparison(self,
                                           clean_saliency: np.ndarray,
                                           noisy_saliency: np.ndarray,
                                           variance_map: np.ndarray,
                                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create 3-panel figure showing noise sensitivity (Fig 5).
        
        Panels:
        1. Clean saliency map
        2. Noisy saliency map (under σ=0.04 noise)
        3. Variance heatmap (red = brittle regions)
        
        Args:
            clean_saliency: Baseline saliency map (H, W)
            noisy_saliency: Mean saliency under noise (H, W)
            variance_map: Variance across N runs (H, W)
            save_path: Path to save figure (optional)
        
        Returns:
            matplotlib Figure object
        """
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Noise Sensitivity Analysis (σ = 0.04)', fontsize=18, fontweight='bold')
        
        # Panel 1: Clean saliency
        im1 = axes[0].imshow(clean_saliency, cmap='jet', aspect='auto')
        axes[0].set_title('Clean Saliency Map', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Width', fontsize=12)
        axes[0].set_ylabel('Height', fontsize=12)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].grid(False)
        
        # Panel 2: Noisy saliency
        im2 = axes[1].imshow(noisy_saliency, cmap='jet', aspect='auto')
        axes[1].set_title('Noisy Saliency Map\n(Mean over N=20 runs)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Width', fontsize=12)
        axes[1].set_ylabel('Height', fontsize=12)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].grid(False)
        
        # Panel 3: Variance map (red = brittle regions)
        im3 = axes[2].imshow(variance_map, cmap='hot', aspect='auto')
        axes[2].set_title('Variance Heatmap\n(Red = Brittle Regions)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Width', fontsize=12)
        axes[2].set_ylabel('Height', fontsize=12)
        cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Variance', fontsize=10)
        axes[2].grid(False)
        
        plt.tight_layout()
        
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.save_pdf:
                pdf_path = save_path.with_suffix('.pdf')
                save_figure_as_pdf(fig, pdf_path)
            print(f"Saved noise sensitivity comparison to {save_path}")
        
        return fig
    
    def create_prediction_inconsistency_curve(self,
                                            pi_with_dcg: Dict[str, Any],
                                            pi_without_dcg: Optional[Dict[str, Any]] = None,
                                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create prediction inconsistency (PI) curve (Fig 6).
        
        PI(σ) = (1/N) * Σ [predicted_label_i != predicted_label_clean]
        
        Args:
            pi_with_dcg: Dictionary with 'noise_levels' and 'pi_scores' for DCG model
            pi_without_dcg: Optional dictionary for ablation (if None, only show DCG)
            save_path: Path to save figure (optional)
        
        Returns:
            matplotlib Figure object
        """
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        noise_levels = pi_with_dcg['noise_levels']
        pi_scores_dcg = pi_with_dcg['pi_scores']
        
        # Plot with-DCG curve
        ax.plot(noise_levels, pi_scores_dcg, 
               marker='o', markersize=10, linewidth=2.5, 
               color='#2ca02c', label='With DCG', linestyle='-')
        
        # Plot without-DCG curve if provided
        if pi_without_dcg is not None:
            pi_scores_no_dcg = pi_without_dcg['pi_scores']
            ax.plot(noise_levels, pi_scores_no_dcg,
                   marker='s', markersize=10, linewidth=2.5,
                   color='#d62728', label='Without DCG', linestyle='--')
        
        ax.set_xlabel('Noise Level (σ)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Prediction Inconsistency (PI)', fontsize=14, fontweight='bold')
        ax.set_title('Prediction Inconsistency Under Input Noise', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_xlim([min(noise_levels) - 0.01, max(noise_levels) + 0.01])
        ax.set_ylim([0, max(pi_scores_dcg) * 1.1 if pi_without_dcg is None 
                     else max(max(pi_scores_dcg), max(pi_without_dcg['pi_scores'])) * 1.1])
        
        # Add value annotations
        for i, (sigma, pi) in enumerate(zip(noise_levels, pi_scores_dcg)):
            ax.annotate(f'{pi:.3f}', 
                       xy=(sigma, pi), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=10, 
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        if pi_without_dcg is not None:
            for i, (sigma, pi) in enumerate(zip(noise_levels, pi_without_dcg['pi_scores'])):
                ax.annotate(f'{pi:.3f}',
                           xy=(sigma, pi),
                           xytext=(5, -15),
                           textcoords='offset points',
                           fontsize=10,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.save_pdf:
                pdf_path = save_path.with_suffix('.pdf')
                save_figure_as_pdf(fig, pdf_path)
            print(f"Saved prediction inconsistency curve to {save_path}")
        
        return fig

