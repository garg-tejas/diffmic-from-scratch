"""
Combined Visualizer Module

Creates synchronized multi-panel animations showing the relationship between
denoising, attention mechanisms, and probability evolution.

Features:
- Synchronized animation combining image, attention, and probability bars
- Image denoising sequence with attention overlay
- Side-by-side comparison views
- Metadata panels with timestep info

Purpose:
- Show how different XAI components relate during inference
- Provide comprehensive visual understanding of model behavior
- Enable research into diffusion-based classification dynamics
- Support educational and presentation needs

Future Extensions:
- 3D probability simplex trajectories
- Multi-sample comparison animations
- Interactive WebGL/Three.js animations
- Video format export (MP4, WebM)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.utils.image_utils import create_heatmap_overlay, upsample_attention
from xai.utils.animation_utils import (
    capture_frame, create_subplot_grid, add_timestep_indicator,
    add_prediction_indicator
)
from xai.utils.pdf_utils import save_frame_grid_as_image, save_animation_frames_as_pdf, select_key_frames


class CombinedVisualizer:
    """
    Create synchronized multi-panel animations combining multiple XAI views.
    
    This visualizer creates comprehensive animations that show how different
    aspects of the model's decision-making process evolve together during
    the diffusion denoising process.
    
    Key visualizations:
    1. Synchronized 4-panel animation (image, attention, probabilities, metadata)
    2. Image denoising sequence with attention overlay
    3. Comparative views showing different explanation methods
    
    Attributes:
        config: Configuration dictionary with visualization settings
        class_names: Dictionary mapping class IDs to human-readable names
        colormap: Colormap for attention heatmaps
        alpha: Transparency for attention overlays
    
    Usage:
        >>> visualizer = CombinedVisualizer(config)
        >>> visualizer.create_synchronized_animation(
        ...     image, diffusion_exp, attention_exp, 
        ...     save_path='sync.gif'
        ... )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the combined visualizer.
        
        Args:
            config: Configuration dictionary with keys:
                - colormap: Colormap for heatmaps
                - overlay_alpha: Transparency for overlays
                - class_names: Class ID to name mapping
                - figure_size: Default figure dimensions
                - image_dpi: Resolution for images
        """
        self.config = config
        self.class_names = config.get('class_names', {})
        self.colormap = config.get('colormap', 'jet')
        self.alpha = config.get('overlay_alpha', 0.5)
        self.dpi = config.get('image_dpi', 100)
        
        print(f"[CombinedVisualizer] Initialized with colormap={self.colormap}")
    
    def create_synchronized_animation(self,
                                     image: np.ndarray,
                                     diffusion_explanation: Dict[str, Any],
                                     attention_explanation: Dict[str, Any],
                                     save_path: str,
                                     fps: int = 5) -> str:
        """
        Create synchronized 4-panel animation showing complete denoising process.
        
        This is the flagship visualization showing how all XAI components
        evolve together during the diffusion process.
        
        Layout (2x2 grid):
            Top-left: Original image with evolving attention heatmap
            Top-right: Probability bars for all classes
            Bottom-left: Metadata (timestep, prediction, metrics)
            Bottom-right: Comparison with auxiliary model
        
        Args:
            image: Original input image (H, W, 3)
            diffusion_explanation: Output from DiffusionExplainer with:
                - trajectory: List of states at each timestep
                - attention_evolution: Attention maps through time
            attention_explanation: Output from AttentionExplainer
            save_path: Path to save animated GIF
            fps: Frames per second for animation
        
        Returns:
            Path to saved GIF file
        
        Process:
            1. Extract trajectory and attention evolution
            2. For each timestep:
               - Overlay attention on image
               - Draw probability bars
               - Add metadata panel
               - Add comparison panel
            3. Capture frame and collect
            4. Save as animated GIF
        
        Example:
            >>> vis = CombinedVisualizer(config)
            >>> gif_path = vis.create_synchronized_animation(
            ...     image, diff_exp, att_exp, 'output.gif', fps=5
            ... )
            >>> print(f"Saved to {gif_path}")
        """
        trajectory = diffusion_explanation.get('trajectory', [])
        attention_evolution = diffusion_explanation.get('attention_evolution', [])
        
        if len(trajectory) == 0:
            raise ValueError("No trajectory data found in diffusion explanation")
        
        # Get auxiliary model predictions for comparison
        aux_prediction = attention_explanation.get('prediction', -1)
        aux_confidence = attention_explanation.get('confidence', 0.0)
        
        num_frames = len(trajectory)
        num_classes = len(trajectory[0]['probs'])
        
        print(f"[CombinedVisualizer] Creating synchronized animation with {num_frames} frames...")
        
        # Prepare class labels and colors
        class_labels = [self.class_names.get(str(i), f"Class {i}") for i in range(num_classes)]
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        
        frames = []
        prev_prediction = -1
        
        # Use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        for frame_idx in range(num_frames):
            # Create 2x2 grid
            fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Get data for this timestep
            step_data = trajectory[frame_idx]
            timestep = step_data['timestep']
            probs = step_data['probs']
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])
            
            # Check if prediction changed
            prediction_changed = (prediction != prev_prediction) and (prev_prediction != -1)
            prev_prediction = prediction
            
            # --- Top-left: Image with attention overlay ---
            ax_image = fig.add_subplot(gs[0, 0])
            ax_image.axis('off')
            
            # Get attention for this timestep
            if frame_idx < len(attention_evolution) and attention_evolution[frame_idx]:
                attn_data = attention_evolution[frame_idx]
                
                # Extract attention map
                if 'class_specific_saliency' in attn_data and attn_data['class_specific_saliency'] is not None:
                    saliency = attn_data['class_specific_saliency']
                elif 'spatial_probs' in attn_data and attn_data['spatial_probs'] is not None:
                    spatial_probs = attn_data['spatial_probs']
                    if isinstance(spatial_probs, np.ndarray) and spatial_probs.ndim == 3:
                        saliency = np.max(spatial_probs, axis=0)
                    else:
                        saliency = None
                else:
                    saliency = None
                
                if saliency is not None:
                    # Create attention overlay
                    saliency_up = upsample_attention(saliency, image.shape[:2])
                    overlay = create_heatmap_overlay(image, saliency_up, self.colormap, self.alpha)
                    ax_image.imshow(overlay)
                else:
                    ax_image.imshow(image)
            else:
                ax_image.imshow(image)
            
            ax_image.set_title(f'Attention Evolution - Timestep {timestep}', 
                             fontsize=12, fontweight='bold')
            
            # --- Top-right: Probability bars ---
            ax_probs = fig.add_subplot(gs[0, 1])
            
            # Color bars differently for predicted class
            bar_colors = [colors[i] if i != prediction else 'lightcoral' for i in range(num_classes)]
            bars = ax_probs.bar(range(num_classes), probs, color=bar_colors, 
                               edgecolor='black', linewidth=1)
            
            # Highlight predicted class
            bars[prediction].set_edgecolor('red')
            bars[prediction].set_linewidth(3)
            
            ax_probs.set_xticks(range(num_classes))
            ax_probs.set_xticklabels(class_labels, rotation=45, ha='right')
            ax_probs.set_ylabel('Probability', fontsize=11, fontweight='bold')
            ax_probs.set_title('Class Probabilities', fontsize=12, fontweight='bold')
            ax_probs.set_ylim(0, 1)
            ax_probs.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                if prob > 0.05:
                    height = bar.get_height()
                    ax_probs.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
            
            # --- Bottom-left: Metadata panel ---
            ax_meta = fig.add_subplot(gs[1, 0])
            ax_meta.axis('off')
            
            # Calculate metrics
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            stability = diffusion_explanation.get('stability_score', 0.0)
            
            # Create metadata text
            meta_text = f"""
DENOISING PROGRESS
{'='*40}

Timestep: {timestep}
Frame: {frame_idx + 1}/{num_frames}
Progress: {(frame_idx + 1) / num_frames * 100:.1f}%

CURRENT PREDICTION
{'='*40}

Class: {class_labels[prediction]}
Confidence: {confidence:.1%}
Status: {'CHANGED' if prediction_changed else 'Stable'}

METRICS
{'='*40}

Entropy: {entropy:.3f}
Stability Score: {stability:.3f}
            """
            
            ax_meta.text(0.1, 0.9, meta_text, transform=ax_meta.transAxes,
                        fontsize=10, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            # --- Bottom-right: Comparison panel ---
            ax_compare = fig.add_subplot(gs[1, 1])
            ax_compare.axis('off')
            
            # Compare diffusion vs auxiliary predictions
            compare_text = f"""
MODEL COMPARISON
{'='*40}

Auxiliary Model (DCG):
  Prediction: {class_labels[aux_prediction]}
  Confidence: {aux_confidence:.1%}

Diffusion Model (Current):
  Prediction: {class_labels[prediction]}
  Confidence: {confidence:.1%}

Agreement: {'YES' if prediction == aux_prediction else 'NO'}

{'='*40}

The diffusion model refines the auxiliary
model's prediction through {len(trajectory)} 
denoising steps.
            """
            
            ax_compare.text(0.1, 0.9, compare_text, transform=ax_compare.transAxes,
                          fontsize=10, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
            
            # Add overall title
            fig.suptitle(f'Synchronized Diffusion Analysis', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Adjust layout using subplots_adjust to avoid tight_layout warnings with GridSpec
            fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, hspace=0.3, wspace=0.3)
            
            # Capture frame
            frame = capture_frame(fig)
            frames.append(frame)
            plt.close(fig)
        
        if len(frames) == 0:
            raise ValueError("No frames generated for synchronized visualization")
        
        layout = (5, 2)
        max_panels = layout[0] * layout[1]
        frame_indices = select_key_frames(list(range(len(frames))), max_panels)
        frame_indices = sorted(frame_indices)
        selected_frames = [frames[i] for i in frame_indices]
        
        frame_titles = [f"Frame {i + 1}" for i in range(len(selected_frames))]
        
        if save_path is None:
            save_path = 'synchronized_frames.png'
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        save_frame_grid_as_image(
            selected_frames,
            save_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Synchronized Diffusion Analysis',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[CombinedVisualizer] Stacked synchronized frames saved to {save_path}")
        
        pdf_path = save_path.with_suffix('.pdf')
        save_animation_frames_as_pdf(
            selected_frames,
            pdf_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Synchronized Diffusion Analysis',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[CombinedVisualizer] PDF with synchronized frames saved to {pdf_path}")
        
        return save_path
    
    def create_image_denoising_sequence(self,
                                       image: np.ndarray,
                                       diffusion_explanation: Dict[str, Any],
                                       save_path: str,
                                       fps: int = 5) -> str:
        """
        Create animation showing image with evolving attention overlay.
        
        This animation focuses on the visual aspect: how attention
        regions change as the model refines its prediction through
        the denoising process.
        
        Args:
            image: Original input image (H, W, 3)
            diffusion_explanation: Output from DiffusionExplainer
            save_path: Path to save GIF
            fps: Frames per second
        
        Returns:
            Path to saved GIF file
        
        Visualization:
            - Original image with attention heatmap overlay
            - Overlay intensity/pattern changes each frame
            - Title shows timestep and predicted class
            - Clean, focused view for understanding attention evolution
        
        Use case:
            Perfect for showing how the model's "focus" shifts during
            inference, helping understand which regions are important
            at different stages of the denoising process.
        
        Example:
            >>> vis = CombinedVisualizer(config)
            >>> vis.create_image_denoising_sequence(
            ...     image, diff_exp, 'denoising_seq.gif'
            ... )
        """
        trajectory = diffusion_explanation.get('trajectory', [])
        attention_evolution = diffusion_explanation.get('attention_evolution', [])
        
        if len(trajectory) == 0:
            raise ValueError("No trajectory data")
        
        num_frames = len(trajectory)
        print(f"[CombinedVisualizer] Creating image denoising sequence with {num_frames} frames...")
        
        frames = []
        
        # Use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        for frame_idx in range(num_frames):
            step_data = trajectory[frame_idx]
            timestep = step_data['timestep']
            probs = step_data['probs']
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
            ax.axis('off')
            
            # Get attention for this timestep
            if frame_idx < len(attention_evolution) and attention_evolution[frame_idx]:
                attn_data = attention_evolution[frame_idx]
                
                # Extract attention map
                saliency = None
                if 'class_specific_saliency' in attn_data and attn_data['class_specific_saliency'] is not None:
                    saliency = attn_data['class_specific_saliency']
                elif 'spatial_probs' in attn_data and attn_data['spatial_probs'] is not None:
                    spatial_probs = attn_data['spatial_probs']
                    if isinstance(spatial_probs, np.ndarray) and spatial_probs.ndim == 3:
                        saliency = np.max(spatial_probs, axis=0)
                
                if saliency is not None:
                    # Create attention overlay
                    saliency_up = upsample_attention(saliency, image.shape[:2])
                    overlay = create_heatmap_overlay(image, saliency_up, self.colormap, self.alpha)
                    ax.imshow(overlay)
                else:
                    ax.imshow(image)
            else:
                ax.imshow(image)
            
            # Add title with info
            class_name = self.class_names.get(str(prediction), f"Class {prediction}")
            title = f'Denoising Sequence\n'
            title += f'Timestep: {timestep} | Frame: {frame_idx+1}/{num_frames}\n'
            title += f'Predicted: {class_name} (Confidence: {confidence:.1%})'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            
            # Adjust layout - no need for tight_layout with single subplot
            fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
            
            # Capture frame
            frame = capture_frame(fig)
            frames.append(frame)
            plt.close(fig)
        
        if len(frames) == 0:
            raise ValueError("No frames generated for denoising sequence")
        
        layout = (5, 2)
        max_panels = layout[0] * layout[1]
        frame_indices = select_key_frames(list(range(len(frames))), max_panels)
        frame_indices = sorted(frame_indices)
        selected_frames = [frames[i] for i in frame_indices]
        
        frame_titles = []
        for idx in frame_indices:
            step_data = trajectory[idx]
            timestep = step_data['timestep']
            probs = step_data['probs']
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])
            class_name = self.class_names.get(str(prediction), f"Class {prediction}")
            frame_titles.append(f"t={timestep}, {class_name} ({confidence:.2f})")
        
        if save_path is None:
            save_path = 'denoising_sequence.png'
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        save_frame_grid_as_image(
            selected_frames,
            save_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Image Denoising Sequence',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[CombinedVisualizer] Stacked denoising frames saved to {save_path}")
        
        pdf_path = save_path.with_suffix('.pdf')
        save_animation_frames_as_pdf(
            selected_frames,
            pdf_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Image Denoising Sequence',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[CombinedVisualizer] PDF with denoising frames saved to {pdf_path}")
        
        return save_path


"""
Usage Examples:

# Initialize visualizer
config = {
    'colormap': 'jet',
    'overlay_alpha': 0.5,
    'class_names': {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"},
    'image_dpi': 100
}
visualizer = CombinedVisualizer(config)

# Create synchronized animation
sync_path = visualizer.create_synchronized_animation(
    image=image_array,
    diffusion_explanation=diffusion_exp,
    attention_explanation=attention_exp,
    save_path='outputs/synchronized.gif',
    fps=5
)

# Create image denoising sequence
seq_path = visualizer.create_image_denoising_sequence(
    image=image_array,
    diffusion_explanation=diffusion_exp,
    save_path='outputs/denoising_sequence.gif',
    fps=5
)

print(f"Created animations:")
print(f"  - Synchronized: {sync_path}")
print(f"  - Image sequence: {seq_path}")
"""

