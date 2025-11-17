"""
Guidance Map Visualization Module

Creates visualizations for dense guidance maps showing the interpolation
between global and local priors.

Features:
- Per-class guidance heatmaps showing spatial distribution
- Prior interpolation curves (global → local transition)
- Global vs local prior comparison bar charts
- Distance matrix visualization
- Guidance evolution animations through diffusion timesteps
- Multi-class grid layouts for comprehensive analysis

Purpose:
- Visualize how guidance signals are distributed spatially
- Understand interpolation mechanisms between global and local priors
- Compare predictions from different pathways
- Support clinical validation and model debugging
- Enable research on multi-scale feature fusion

Future Extensions:
- Interactive 3D guidance map visualization
- Guidance map clustering and pattern recognition
- Real-time guidance evolution during inference
- Comparison across multiple samples
- Guidance map animation with probability overlay
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


class GuidanceMapVisualizer:
    """
    Visualize dense guidance maps and prior interpolation.
    
    Creates visualizations showing:
    - Per-class guidance distribution heatmaps
    - Interpolation curves from global to local
    - Global vs local prediction comparison
    - Spatial guidance grid
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the guidance map visualizer.
        
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
        
        # Default class names if not provided
        if not self.class_names:
            self.class_names = {str(i): f"Class {i}" for i in range(10)}
    
    def create_guidance_heatmap(self,
                                guidance_map: np.ndarray,
                                class_idx: Optional[int] = None,
                                save_path: Optional[str] = None) -> np.ndarray:
        """
        Create per-class heatmap showing guidance distribution.
        
        Args:
            guidance_map: Guidance map [C, N_p, N_p] or [N_p, N_p] for single class
            class_idx: If None, show all classes in subplots. If specified, show single class.
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        if guidance_map.ndim == 2:
            # Single class
            guidance_map = guidance_map[np.newaxis, :, :]
            class_idx = 0
        
        num_classes = guidance_map.shape[0]
        
        if class_idx is not None:
            # Single class heatmap
            fig, ax = plt.subplots(figsize=(8, 8))
            heatmap_data = guidance_map[class_idx]
            
            im = ax.imshow(heatmap_data, cmap=self.colormap, aspect='auto')
            ax.set_title(
                f'Guidance Map - {self.class_names.get(str(class_idx), f"Class {class_idx}")}',
                fontsize=14, fontweight='bold'
            )
            ax.set_xlabel('Spatial Location j', fontsize=12)
            ax.set_ylabel('Spatial Location i', fontsize=12)
            plt.colorbar(im, ax=ax, label='Probability')
            
            # Add diagonal line
            n = heatmap_data.shape[0]
            ax.plot([0, n-1], [0, n-1], 'r--', linewidth=2, alpha=0.7, label='Diagonal (Global)')
            ax.legend()
            
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
            
            # Save PDF if enabled
            if self.save_pdf and save_path:
                pdf_path = Path(save_path).with_suffix('.pdf')
                save_figure_as_pdf(fig, pdf_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            vis_array = figure_to_array(fig)
            plt.close(fig)
            
        else:
            # Multi-class grid
            n_cols = min(3, num_classes)
            n_rows = (num_classes + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
            gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
            
            for c in range(num_classes):
                row = c // n_cols
                col = c % n_cols
                ax = fig.add_subplot(gs[row, col])
                
                heatmap_data = guidance_map[c]
                im = ax.imshow(heatmap_data, cmap=self.colormap, aspect='auto')
                ax.set_title(
                    self.class_names.get(str(c), f"Class {c}"),
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel('j', fontsize=10)
                ax.set_ylabel('i', fontsize=10)
                
                # Add diagonal
                n = heatmap_data.shape[0]
                ax.plot([0, n-1], [0, n-1], 'r--', linewidth=1.5, alpha=0.7)
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            fig.suptitle('Guidance Maps by Class', fontsize=16, fontweight='bold', y=0.98)
            fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05)
            
            # Save PDF if enabled
            if self.save_pdf and save_path:
                pdf_path = Path(save_path).with_suffix('.pdf')
                save_figure_as_pdf(fig, pdf_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            vis_array = figure_to_array(fig)
            plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_interpolation_plot(self,
                                  y_global: np.ndarray,
                                  y_local: np.ndarray,
                                  distance_matrix: np.ndarray,
                                  save_path: Optional[str] = None) -> np.ndarray:
        """
        Show smooth transition from global to local priors.
        
        Args:
            y_global: Global prior probabilities [C]
            y_local: Local prior probabilities [C]
            distance_matrix: Distance matrix [N_p, N_p]
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        num_classes = len(y_global)
        n_patches = distance_matrix.shape[0]
        
        # Get unique distances and compute interpolated values
        unique_distances = np.unique(distance_matrix.flatten())
        unique_distances = np.sort(unique_distances)
        
        # Normalize distances to [0, 1]
        max_dist = unique_distances.max()
        normalized_dist = unique_distances / max_dist if max_dist > 0 else unique_distances
        
        # Compute interpolated probabilities for each distance
        interpolated_probs = []
        for dist_norm in normalized_dist:
            # Interpolation: (1 - dist) * global + dist * local
            interp = (1 - dist_norm) * y_global + dist_norm * y_local
            interpolated_probs.append(interp)
        interpolated_probs = np.array(interpolated_probs)  # [num_distances, num_classes]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Interpolation curves for each class
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for c in range(num_classes):
            ax1.plot(normalized_dist, interpolated_probs[:, c],
                    marker='o', label=self.class_names.get(str(c), f"Class {c}"),
                    color=colors[c], linewidth=2, markersize=4)
        
        ax1.set_xlabel('Normalized Distance from Diagonal', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Prior Interpolation: Global → Local', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Add annotations
        ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Global (diagonal)')
        ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Local (off-diagonal)')
        
        # Plot 2: Distance matrix visualization
        im = ax2.imshow(distance_matrix, cmap='coolwarm', aspect='auto')
        ax2.set_title('Distance Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Spatial Location j', fontsize=12)
        ax2.set_ylabel('Spatial Location i', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Distance |i - j|')
        
        # Add diagonal
        n = distance_matrix.shape[0]
        ax2.plot([0, n-1], [0, n-1], 'k--', linewidth=2, alpha=0.7)
        
        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.25)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_prior_comparison(self,
                               y_global: np.ndarray,
                               y_local: np.ndarray,
                               y_fusion: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> np.ndarray:
        """
        Bar chart comparing global vs local predictions.
        
        Args:
            y_global: Global prior probabilities [C]
            y_local: Local prior probabilities [C]
            y_fusion: Fusion probabilities [C] (optional)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        num_classes = len(y_global)
        x = np.arange(num_classes)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, y_global, width, label='Global Prior', alpha=0.8)
        bars2 = ax.bar(x, y_local, width, label='Local Prior', alpha=0.8)
        
        if y_fusion is not None:
            bars3 = ax.bar(x + width, y_fusion, width, label='Fusion', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Global vs Local Prior Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self.class_names.get(str(i), f"Class {i}") for i in range(num_classes)],
                          rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2] + ([bars3] if y_fusion is not None else []):
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_guidance_evolution_animation_from_trajectory(self,
                                                            diffusion_trajectory: List[Dict[str, Any]],
                                                            save_path: Optional[str] = None,
                                                            fps: int = 1) -> str:
        """
        Create guidance evolution animation from diffusion trajectory.
        
        Uses spatial_probs from trajectory to visualize guidance evolution.
        
        Args:
            diffusion_trajectory: List of trajectory steps from DiffusionExplainer
            save_path: Path to save GIF
            fps: Frames per second (default 1 for slow viewing)
            
        Returns:
            Path to saved GIF
        """
        import matplotlib
        matplotlib.use('Agg')
        
        if len(diffusion_trajectory) == 0:
            raise ValueError("No trajectory data provided for animation")
        
        frames = []
        num_frames = len(diffusion_trajectory)
        
        for frame_idx, step in enumerate(diffusion_trajectory):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
            
            timestep = step.get('timestep', frame_idx)
            pred_class = step.get('predicted_class', 0)
            spatial_probs = step.get('spatial_probs', None)
            probs = step.get('probs', None)
            
            # Left panel: Spatial probability heatmap (guidance-like)
            if spatial_probs is not None:
                # spatial_probs is [C, H, W], show for predicted class
                if spatial_probs.ndim == 3:
                    heatmap_data = spatial_probs[pred_class]
                else:
                    heatmap_data = spatial_probs
                
                im1 = ax1.imshow(heatmap_data, cmap=self.colormap, aspect='auto', vmin=0, vmax=1)
                ax1.set_title(f'Spatial Probabilities (Class {pred_class})', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Width', fontsize=10)
                ax1.set_ylabel('Height', fontsize=10)
                plt.colorbar(im1, ax=ax1, label='Probability')
            else:
                ax1.text(0.5, 0.5, 'No spatial data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Spatial Probabilities', fontsize=12)
            
            # Right panel: Class probabilities bar chart
            if probs is not None:
                num_classes = len(probs)
                colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
                bar_colors = [colors[i] if i != pred_class else 'lightcoral' for i in range(num_classes)]
                
                bars = ax2.bar(range(num_classes), probs, color=bar_colors, edgecolor='black', linewidth=1)
                bars[pred_class].set_edgecolor('red')
                bars[pred_class].set_linewidth(3)
                
                ax2.set_xticks(range(num_classes))
                ax2.set_xticklabels([self.class_names.get(str(i), f"C{i}") for i in range(num_classes)],
                                  rotation=45, ha='right', fontsize=9)
                ax2.set_ylabel('Probability', fontsize=11)
                ax2.set_title(f'Class Probabilities', fontsize=12, fontweight='bold')
                ax2.set_ylim(0, 1)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    if prob > 0.05:
                        ax2.text(bar.get_x() + bar.get_width()/2., prob + 0.02,
                               f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
            
            fig.suptitle(f'Guidance Evolution - Timestep: {timestep} | Step: {frame_idx+1}/{num_frames}',
                        fontsize=14, fontweight='bold')
            fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.25)
            
            frame = capture_frame(fig)
            frames.append(frame)
            plt.close(fig)
            
            if (frame_idx + 1) % max(1, num_frames // 10) == 0:
                print(f"  Frame {frame_idx + 1}/{num_frames} captured")
        
        if len(frames) == 0:
            raise ValueError("No frames generated for guidance evolution visualization")
        
        layout = (5, 2)
        max_panels = layout[0] * layout[1]
        frame_indices = select_key_frames(list(range(len(frames))), max_panels)
        frame_indices = sorted(frame_indices)
        selected_frames = [frames[i] for i in frame_indices]
        
        frame_titles = []
        for idx in frame_indices:
            step = diffusion_trajectory[idx]
            timestep = step.get('timestep', idx)
            pred_class = step.get('predicted_class', -1)
            class_name = self.class_names.get(str(pred_class), f"Class {pred_class}") if pred_class != -1 else "Unknown"
            frame_titles.append(f"t={timestep}, {class_name}")
        
        if save_path is None:
            save_path = 'guidance_evolution.png'
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        save_frame_grid_as_image(
            selected_frames,
            save_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Guidance Evolution Through Diffusion Timesteps',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[GuidanceMapVisualizer] Stacked guidance frames saved to {save_path}")
        
        if self.save_pdf:
            pdf_path = save_path.with_suffix('.pdf')
            save_animation_frames_as_pdf(
                selected_frames,
                pdf_path,
                layout=layout,
                titles=frame_titles,
                suptitle='Guidance Evolution Through Diffusion Timesteps',
                dpi=self.dpi,
                frame_size=(3.5, 3.5),
                spacing=0.4
            )
            print(f"[GuidanceMapVisualizer] PDF with frames saved to {pdf_path}")
        
        return save_path
    
    def create_guidance_evolution_animation(self,
                                           guidance_sequence: List[np.ndarray],
                                           timesteps: List[int],
                                           save_path: Optional[str] = None,
                                           fps: int = 1) -> str:
        """
        Animated guidance map evolution during diffusion.
        
        Args:
            guidance_sequence: List of guidance maps [C, N_p, N_p] at each timestep
            timesteps: List of timestep values
            save_path: Path to save GIF
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        import matplotlib
        matplotlib.use('Agg')
        
        if len(guidance_sequence) == 0:
            raise ValueError("No guidance maps provided for animation")
        
        num_classes = guidance_sequence[0].shape[0]
        frames = []
        
        # Use predicted class for visualization (or class 0 if unknown)
        pred_class = 0
        
        for i, guidance_map in enumerate(guidance_sequence):
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
            
            # Show guidance map for predicted class (or max probability class)
            if guidance_map.ndim == 3:
                # Find class with max probability at diagonal (global)
                diag_vals = np.diag(guidance_map.max(axis=1).max(axis=1))
                pred_class = int(np.argmax(diag_vals))
                heatmap_data = guidance_map[pred_class]
            else:
                heatmap_data = guidance_map
            
            im = ax.imshow(heatmap_data, cmap=self.colormap, aspect='auto', vmin=0, vmax=1)
            ax.set_title(
                f'Guidance Evolution - Timestep: {timesteps[i] if i < len(timesteps) else i}\n'
                f'Class: {self.class_names.get(str(pred_class), f"Class {pred_class}")}',
                fontsize=12, fontweight='bold'
            )
            ax.set_xlabel('Spatial Location j', fontsize=11)
            ax.set_ylabel('Spatial Location i', fontsize=11)
            plt.colorbar(im, ax=ax, label='Probability')
            
            # Add diagonal
            n = heatmap_data.shape[0]
            ax.plot([0, n-1], [0, n-1], 'r--', linewidth=2, alpha=0.7)
            
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
            
            frame = capture_frame(fig)
            frames.append(frame)
            plt.close(fig)
            
            if (i + 1) % max(1, len(guidance_sequence) // 10) == 0:
                print(f"  Frame {i + 1}/{len(guidance_sequence)} captured")
        
        if len(frames) == 0:
            raise ValueError("No frames generated for guidance evolution visualization")
        
        layout = (5, 2)
        max_panels = layout[0] * layout[1]
        frame_indices = select_key_frames(list(range(len(frames))), max_panels)
        frame_indices = sorted(frame_indices)
        selected_frames = [frames[i] for i in frame_indices]
        
        frame_titles = []
        for idx in frame_indices:
            timestep = timesteps[idx] if idx < len(timesteps) else idx
            frame_titles.append(f"t={timestep}")
        
        if save_path is None:
            save_path = 'guidance_evolution.png'
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        save_frame_grid_as_image(
            selected_frames,
            save_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Guidance Evolution Heatmaps',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[GuidanceMapVisualizer] Stacked guidance frames saved to {save_path}")
        
        if self.save_pdf:
            pdf_path = save_path.with_suffix('.pdf')
            save_animation_frames_as_pdf(
                selected_frames,
                pdf_path,
                layout=layout,
                titles=frame_titles,
                suptitle='Guidance Evolution Heatmaps',
                dpi=self.dpi,
                frame_size=(3.5, 3.5),
                spacing=0.4
            )
            print(f"[GuidanceMapVisualizer] PDF with frames saved to {pdf_path}")
        
        return save_path

