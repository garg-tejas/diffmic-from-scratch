"""
Spatio-Temporal Visualizer Module

This module creates visualizations for spatio-temporal attention evolution,
showing how attention shifts from global to local features over timesteps.

Purpose:
- Plot attention evolution over timesteps
- Create animated heatmaps showing attention shifts
- Visualize coarse-to-fine transition
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SpatioTemporalVisualizer:
    """
    Visualize spatio-temporal attention evolution.
    
    Attributes:
        config: Visualization configuration dictionary
        
    Usage:
        >>> vis = SpatioTemporalVisualizer(config)
        >>> vis.plot_attention_evolution(attention_trajectory, 'output.png')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spatio-temporal visualizer.
        
        Args:
            config: Configuration with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'jet')
        self.figure_size = config.get('figure_size', [12, 8])
        self.dpi = config.get('image_dpi', 300)
    
    def plot_attention_evolution(self,
                                 attention_trajectory: List[Dict],
                                 save_path: Path,
                                 title: Optional[str] = None) -> None:
        """
        Plot line chart showing attention evolution over timesteps.
        
        Args:
            attention_trajectory: List of attention data at each timestep
            save_path: Path to save figure
            title: Optional title for the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
        
        timesteps = [step['timestep'] for step in attention_trajectory]
        global_attn = [step['global_attention'] for step in attention_trajectory]
        local_attn = [step['local_attention'] for step in attention_trajectory]
        
        # Plot global vs local
        axes[0].plot(timesteps, global_attn, 'b-', linewidth=2, label='Global (F_raw)', marker='o')
        axes[0].plot(timesteps, local_attn, 'r-', linewidth=2, label='Local (F_rois)', marker='s')
        axes[0].set_ylabel('Attention Weight', fontsize=12)
        axes[0].set_title('Global vs Local Attention Evolution', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        
        # Plot per-ROI attention
        if len(attention_trajectory) > 0 and 'roi_attention' in attention_trajectory[0]:
            num_rois = len(attention_trajectory[0]['roi_attention'])
            roi_attn_data = [[step['roi_attention'][i] for step in attention_trajectory] 
                            for i in range(num_rois)]
            
            for i, roi_attn in enumerate(roi_attn_data):
                axes[1].plot(timesteps, roi_attn, linewidth=2, label=f'ROI {i+1}', marker='o', alpha=0.7)
        
        axes[1].set_xlabel('Timestep', fontsize=12)
        axes[1].set_ylabel('Attention Weight', fontsize=12)
        axes[1].set_title('Per-ROI Attention Evolution', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim([0, max([max(roi_attn) for roi_attn in roi_attn_data]) * 1.1] if roi_attn_data else [0, 1])
        
        # Invert x-axis (high timestep = high noise, low timestep = low noise)
        axes[0].invert_xaxis()
        axes[1].invert_xaxis()
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_attention_heatmap_sequence(self,
                                         attention_trajectory: List[Dict],
                                         save_path: Path,
                                         fps: int = 2) -> None:
        """
        Create animated heatmap sequence showing attention evolution.
        
        Args:
            attention_trajectory: List of attention data at each timestep
            save_path: Path to save GIF
            fps: Frames per second
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data: create 7-component attention vector for each timestep
        # [global, roi1, roi2, ..., roi6]
        attention_matrix = []
        timesteps = []
        
        for step in attention_trajectory:
            global_attn = step['global_attention']
            roi_attn = step['roi_attention']
            attention_vector = [global_attn] + roi_attn
            attention_matrix.append(attention_vector)
            timesteps.append(step['timestep'])
        
        attention_matrix = np.array(attention_matrix).T  # (7, num_timesteps)
        
        def animate(frame):
            ax.clear()
            
            # Plot heatmap up to current frame
            current_data = attention_matrix[:, :frame+1]
            im = ax.imshow(current_data, cmap=self.colormap, aspect='auto', vmin=0, vmax=1)
            
            # Labels
            component_labels = ['Global'] + [f'ROI {i+1}' for i in range(6)]
            ax.set_yticks(range(len(component_labels)))
            ax.set_yticklabels(component_labels)
            
            if frame < len(timesteps):
                ax.set_title(f'Attention Evolution (t={timesteps[frame]})', fontsize=14)
            else:
                ax.set_title('Attention Evolution (Complete)', fontsize=14)
            
            ax.set_xlabel('Timestep Index', fontsize=12)
            ax.set_ylabel('Feature Component', fontsize=12)
            
            # Add colorbar
            if frame == 0:
                plt.colorbar(im, ax=ax, label='Attention Weight')
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(attention_trajectory), 
            interval=1000/fps, repeat=True
        )
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
    
    def plot_coarse_to_fine_transition(self,
                                      attention_trajectory: List[Dict],
                                      save_path: Path) -> None:
        """
        Plot showing the coarse-to-fine transition (global → local).
        
        Args:
            attention_trajectory: List of attention data at each timestep
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        timesteps = [step['timestep'] for step in attention_trajectory]
        global_attn = [step['global_attention'] for step in attention_trajectory]
        local_attn = [step['local_attention'] for step in attention_trajectory]
        
        # Plot with area fill
        ax.fill_between(timesteps, 0, global_attn, alpha=0.5, color='blue', label='Global')
        ax.fill_between(timesteps, global_attn, [g + l for g, l in zip(global_attn, local_attn)], 
                       alpha=0.5, color='red', label='Local')
        
        ax.plot(timesteps, global_attn, 'b-', linewidth=2)
        ax.plot(timesteps, [g + l for g, l in zip(global_attn, local_attn)], 'r-', linewidth=2)
        
        ax.set_xlabel('Timestep (High Noise → Low Noise)', fontsize=12)
        ax.set_ylabel('Cumulative Attention', fontsize=12)
        ax.set_title('Coarse-to-Fine Transition', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.invert_xaxis()  # High timestep (high noise) on left
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

