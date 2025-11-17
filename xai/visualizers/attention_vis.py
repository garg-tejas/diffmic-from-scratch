"""
Attention Visualization Module

This module creates visualizations from attention-based explanations.
It takes attention artifacts (saliency maps, patch locations, attention weights)
and generates human-interpretable visualizations.

Visualization Types:
1. Saliency Overlay: Heatmap overlaid on original image
2. Patch Highlighting: Bounding boxes showing attended patches
3. Multi-scale View: Combined global + local attention
4. Class-specific Attention: Separate maps for each class
5. Comparison View: Multiple visualizations side-by-side

Purpose:
- Make attention mechanisms interpretable
- Highlight what the model focuses on
- Support clinical validation and debugging

Future Extensions:
- Interactive visualizations (hover for values)
- 3D visualization for volumetric data
- Animation showing attention evolution
- Comparison with human expert attention
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image
import json
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.utils.image_utils import (
    create_heatmap_overlay,
    draw_bounding_boxes,
    upsample_attention,
    save_visualization,
    create_multi_panel_figure,
    normalize_to_uint8
)
from xai.utils.pdf_utils import (
    save_figure_as_pdf,
    save_animation_frames_as_pdf,
    save_frame_grid_as_image,
    select_key_frames,
    apply_publication_style
)


class AttentionVisualizer:
    """
    Generate visual explanations from attention data.
    
    This class takes the output from AttentionExplainer and creates
    publication-quality visualizations suitable for:
    - Research papers
    - Clinical presentations
    - Model debugging
    - Educational purposes
    
    Attributes:
        config: Configuration dictionary with visualization settings
        colormap: Colormap for heatmaps
        alpha: Opacity for overlays
        
    Usage:
        >>> visualizer = AttentionVisualizer(config)
        >>> visualizer.visualize_attention(
        ...     image, explanation, save_path='output.png'
        ... )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration with visualization settings:
                - colormap: Colormap name ('jet', 'viridis', etc.)
                - overlay_alpha: Opacity for overlays (0-1)
                - figure_size: Default figure size
                - dpi: Image resolution
        """
        self.config = config
        self.colormap = config.get('colormap', 'jet')
        self.alpha = config.get('overlay_alpha', 0.5)
        self.figsize = tuple(config.get('figure_size', [12, 8]))
        self.dpi = config.get('image_dpi', 300)
        self.class_names = config.get('class_names', {})
        self.save_pdf = config.get('save_pdf', True)
        
        # Apply publication style
        apply_publication_style()
        
        print(f"[AttentionVisualizer] Initialized")
        print(f"  Colormap: {self.colormap}")
        print(f"  Alpha: {self.alpha}")
        print(f"  PDF export: {self.save_pdf}")
    
    def visualize_attention(self,
                           image: Union[np.ndarray, Image.Image],
                           explanation: Dict[str, Any],
                           save_path: Optional[str] = None,
                           show_patches: bool = True,
                           show_weights: bool = True) -> np.ndarray:
        """
        Create comprehensive attention visualization.
        
        Args:
            image: Original image
            explanation: Explanation dict from AttentionExplainer
            save_path: Path to save visualization (optional)
            show_patches: Whether to draw patch bounding boxes
            show_weights: Whether to show patch attention weights
        
        Returns:
            Visualization as numpy array
        
        Layout:
            [Original] [Saliency Overlay] [Patches] [Class-specific]
        """
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Extract attention components
        saliency_map = explanation['saliency_map']  # May be (1, num_classes, H, W) or (num_classes, H, W)
        patch_locations = explanation['patch_locations']  # May be (1, num_patches, 2) or (num_patches, 2)
        patch_attention = explanation['patch_attention']  # May be (1, num_patches) or (num_patches,)
        prediction = explanation['prediction']
        
        # Remove batch dimension if present
        if saliency_map.ndim == 4:
            saliency_map = saliency_map[0]  # (num_classes, H, W)
        if patch_locations.ndim == 3:
            patch_locations = patch_locations[0]  # (num_patches, 2)
        if patch_attention.ndim == 2:
            patch_attention = patch_attention[0]  # (num_patches,)
        
        # Create figure with improved layout for publication
        fig = plt.figure(figsize=(16, 4.5))
        gs = GridSpec(1, 4, figure=fig, 
                     left=0.05, right=0.95, top=0.88, bottom=0.12,
                     wspace=0.15, hspace=0.1)
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(image)
        ax1.set_title('(a) Original Image', fontsize=11, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # 2. Saliency overlay (class-averaged)
        ax2 = fig.add_subplot(gs[1])
        saliency_avg = np.mean(saliency_map, axis=0)  # Average across classes -> (H, W)
        saliency_up = upsample_attention(saliency_avg, image.shape[:2])
        overlay = create_heatmap_overlay(image, saliency_up, self.colormap, self.alpha)
        ax2.imshow(overlay)
        ax2.set_title('(b) Saliency Overlay', fontsize=11, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # 3. Patches highlighted
        ax3 = fig.add_subplot(gs[2])
        if show_patches and patch_locations is not None:
            # Convert patch locations to bounding boxes
            boxes = self._patch_locations_to_boxes(
                patch_locations, image.shape, crop_size=(73, 73)
            )
            
            # Create labels with attention weights
            if show_weights:
                labels = [f"{weight:.2f}" for weight in patch_attention]
            else:
                labels = None
            
            img_with_boxes = draw_bounding_boxes(
                image, boxes, labels=labels,
                thickness=2, font_scale=0.7
            )
            ax3.imshow(img_with_boxes)
            ax3.set_title(f'(c) Attention Patches (n={len(boxes)})', 
                         fontsize=11, fontweight='bold', pad=10)
        else:
            ax3.imshow(image)
            ax3.set_title('(c) No Patch Data', fontsize=11, pad=10)
        ax3.axis('off')
        
        # 4. Class-specific saliency for predicted class
        ax4 = fig.add_subplot(gs[3])
        # saliency_map is now (num_classes, H, W) after batch removal
        class_saliency = saliency_map[prediction]  # (H, W)
        class_saliency_up = upsample_attention(class_saliency, image.shape[:2])
        class_overlay = create_heatmap_overlay(
            image, class_saliency_up, self.colormap, self.alpha
        )
        ax4.imshow(class_overlay)
        class_name = self.class_names.get(str(prediction), f"Class {prediction}")
        ax4.set_title(f'(d) {class_name} Saliency', fontsize=11, fontweight='bold', pad=10)
        ax4.axis('off')
        
        # Add overall title with better formatting
        confidence = explanation['confidence']
        class_name = self.class_names.get(str(prediction), f"Class {prediction}")
        fig.suptitle(
            f'Attention Visualization: {class_name} (Confidence: {confidence:.3f})',
            fontsize=13, fontweight='bold', y=0.96
        )
        
        # Convert to array for PNG export
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3]
        
        # Save PNG if requested
        if save_path:
            save_visualization(vis_array, save_path, self.dpi)
        
        # Save PDF version if enabled
        if self.save_pdf and save_path:
            pdf_path = Path(save_path).with_suffix('.pdf')
            save_figure_as_pdf(fig, pdf_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
        
        plt.close(fig)
        
        return vis_array
    
    def visualize_all_class_saliency(self,
                                     image: Union[np.ndarray, Image.Image],
                                     saliency_map: np.ndarray,
                                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize saliency maps for all classes.
        
        Args:
            image: Original image
            saliency_map: Saliency map (num_classes, H, W)
            save_path: Path to save
        
        Returns:
            Visualization array
        
        Use Case:
            Understanding what features indicate each class.
            E.g., "What suggests Moderate DR vs Severe DR?"
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Remove batch dimension if present
        if saliency_map.ndim == 4:
            saliency_map = saliency_map[0]  # (num_classes, H, W)
        
        num_classes = saliency_map.shape[0]
        
        # Create grid
        ncols = min(num_classes, 3)
        nrows = (num_classes + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        if num_classes == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        
        for i in range(num_classes):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            
            # Upsample and overlay
            class_sal = saliency_map[i]
            class_sal_up = upsample_attention(class_sal, image.shape[:2])
            overlay = create_heatmap_overlay(image, class_sal_up, self.colormap, self.alpha)
            
            ax.imshow(overlay)
            class_name = self.class_names.get(str(i), f"Class {i}")
            ax.set_title(class_name, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Hide extra subplots
        for i in range(num_classes, nrows * ncols):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle('Class-Specific Saliency Maps', fontsize=16, fontweight='bold')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.3)
        
        # Convert to array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        
        if save_path:
            save_visualization(vis_array, save_path, self.dpi)
        
        return vis_array
    
    def visualize_patch_attention_bars(self,
                                       patch_attention: np.ndarray,
                                       save_path: Optional[str] = None) -> np.ndarray:
        """
        Create bar chart of patch attention weights.
        
        Args:
            patch_attention: Attention weights (num_patches,)
            save_path: Save path
        
        Returns:
            Visualization array
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_patches = len(patch_attention)
        x = np.arange(num_patches)
        
        # Create bars
        bars = ax.bar(x, patch_attention, color='steelblue', edgecolor='black', linewidth=1.5)
        
        # Highlight most important patch
        max_idx = np.argmax(patch_attention)
        bars[max_idx].set_color('coral')
        
        ax.set_xlabel('Patch Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title('Patch Attention Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{i}' for i in range(num_patches)])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(zip(x, patch_attention)):
            ax.text(idx, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.3)
        
        # Convert to array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        
        if save_path:
            save_visualization(vis_array, save_path, self.dpi)
        
        return vis_array
    
    def visualize_global_local_comparison(self,
                                          image: Union[np.ndarray, Image.Image],
                                          explanation: Dict[str, Any],
                                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Compare global vs local pathway predictions.
        
        Args:
            image: Original image
            explanation: Explanation dict
            save_path: Save path
        
        Returns:
            Visualization array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        global_pred = explanation['global_prediction']
        local_pred = explanation['local_prediction']
        fusion_pred = explanation['fusion_prediction']
        
        # Softmax to get probabilities
        global_probs = np.exp(global_pred) / np.sum(np.exp(global_pred))
        local_probs = np.exp(local_pred) / np.sum(np.exp(local_pred))
        fusion_probs = np.exp(fusion_pred) / np.sum(np.exp(fusion_pred))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        class_labels = [self.class_names.get(str(i), f"C{i}") for i in range(len(global_probs))]
        x = np.arange(len(class_labels))
        width = 0.25
        
        # Global
        axes[0].bar(x, global_probs, width, label='Global', color='steelblue')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Probability', fontsize=12)
        axes[0].set_title('Global Pathway', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_labels, rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Local
        axes[1].bar(x, local_probs, width, label='Local', color='coral')
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Probability', fontsize=12)
        axes[1].set_title('Local Pathway', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_labels, rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Fusion (comparison)
        axes[2].bar(x - width, global_probs, width, label='Global', color='steelblue', alpha=0.7)
        axes[2].bar(x, local_probs, width, label='Local', color='coral', alpha=0.7)
        axes[2].bar(x + width, fusion_probs, width, label='Fusion', color='green', alpha=0.7)
        axes[2].set_xlabel('Class', fontsize=12)
        axes[2].set_ylabel('Probability', fontsize=12)
        axes[2].set_title('Comparison', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(class_labels, rotation=45)
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Global vs Local Pathway Predictions', fontsize=16, fontweight='bold')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.3)
        
        # Convert to array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        
        if save_path:
            save_visualization(vis_array, save_path, self.dpi)
        
        return vis_array
    
    def _patch_locations_to_boxes(self,
                                  patch_locations: np.ndarray,
                                  image_shape: Tuple[int, int],
                                  crop_size: Tuple[int, int] = (73, 73)) -> List[Tuple]:
        """
        Convert patch center locations to bounding boxes.
        
        Args:
            patch_locations: Array of (x, y) coordinates (num_patches, 2)
            image_shape: (height, width) of image
            crop_size: (crop_height, crop_width) of each patch
        
        Returns:
            List of (x, y, width, height) tuples
        """
        boxes = []
        crop_h, crop_w = crop_size
        
        for loc in patch_locations:
            # loc is center point
            center_x, center_y = loc
            
            # Calculate box corners
            x = int(center_x - crop_w // 2)
            y = int(center_y - crop_h // 2)
            
            # Clip to image bounds
            x = max(0, min(x, image_shape[1] - crop_w))
            y = max(0, min(y, image_shape[0] - crop_h))
            
            boxes.append((x, y, crop_w, crop_h))
        
        return boxes
    
    def create_interactive_visualization(self,
                                        image: Union[np.ndarray, Image.Image],
                                        explanation: Dict[str, Any],
                                        save_path: Optional[str] = None) -> str:
        """
        Create interactive HTML visualization with hover effects.
        
        Args:
            image: Original image
            explanation: Explanation dict from AttentionExplainer
            save_path: Path to save HTML file
        
        Returns:
            Path to saved HTML file
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        saliency_map = explanation['saliency_map']
        if saliency_map.ndim == 4:
            saliency_map = saliency_map[0]
        
        img_pil = Image.fromarray(image.astype(np.uint8))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        num_classes = saliency_map.shape[0]
        class_names = [self.class_names.get(str(i), f"Class {i}") for i in range(num_classes)]
        
        avg_saliency = np.mean(saliency_map, axis=0)
        saliency_up = upsample_attention(avg_saliency, image.shape[:2])
        saliency_normalized = ((saliency_up - saliency_up.min()) / 
                              (saliency_up.max() - saliency_up.min() + 1e-8) * 255).astype(np.uint8)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Attention Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        #attentionCanvas {{ cursor: crosshair; border: 2px solid #333; border-radius: 5px; }}
        .class-selector button {{ margin: 5px; padding: 8px 15px; border: none; border-radius: 5px; background: #4CAF50; color: white; cursor: pointer; }}
        .class-selector button:hover {{ background: #45a049; }}
        #hoverInfo {{ position: absolute; background: rgba(0,0,0,0.8); color: white; padding: 10px; border-radius: 5px; pointer-events: none; display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Attention Visualization</h1>
        <div style="position: relative; display: inline-block;">
            <canvas id="attentionCanvas" width="{image.shape[1]}" height="{image.shape[0]}"></canvas>
            <div id="hoverInfo"></div>
        </div>
        <div class="class-selector">
            <strong>View:</strong>
            <button onclick="showAverage()">Average</button>
            {''.join([f'<button onclick="showClass({i})">{class_names[i]}</button>' for i in range(num_classes)])}
        </div>
    </div>
    <script>
        const img = new Image();
        img.src = 'data:image/png;base64,{img_base64}';
        const canvas = document.getElementById('attentionCanvas');
        const ctx = canvas.getContext('2d');
        img.onload = function() {{ ctx.drawImage(img, 0, 0); }};
        canvas.addEventListener('mousemove', function(e) {{
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor(e.clientX - rect.left);
            const y = Math.floor(e.clientY - rect.top);
            const info = document.getElementById('hoverInfo');
            if (x >= 0 && x < {image.shape[1]} && y >= 0 && y < {image.shape[0]}) {{
                info.style.display = 'block';
                info.style.left = (e.clientX + 10) + 'px';
                info.style.top = (e.clientY + 10) + 'px';
                info.innerHTML = 'Position: (' + x + ', ' + y + ')';
            }}
        }});
        function showAverage() {{}}
        {''.join([f'function showClass({i}) {{}}' for i in range(num_classes)])}
    </script>
</body>
</html>"""
        
        if save_path is None:
            save_path = 'interactive_attention.html'
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"[AttentionVisualizer] Interactive visualization saved to {save_path}")
        return save_path
    
    def create_attention_evolution_animation(self,
                                           image: Union[np.ndarray, Image.Image],
                                           attention_sequence: List[Dict[str, Any]],
                                           save_path: Optional[str] = None,
                                           fps: int = 2) -> str:
        """
        Create animation showing attention evolution over time.
        
        Args:
            image: Original image
            attention_sequence: List of attention maps over time
            save_path: Path to save GIF
            fps: Frames per second
        
        Returns:
            Path to saved GIF file
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        frames = []
        num_steps = len(attention_sequence)
        
        # Use non-interactive backend for better reliability
        import matplotlib
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Non-interactive backend
        
        for i, attn_data in enumerate(attention_sequence):
            # Get attention map for this timestep
            saliency = None
            if 'class_specific_saliency' in attn_data and attn_data['class_specific_saliency'] is not None:
                saliency = attn_data['class_specific_saliency']
            elif 'spatial_probs' in attn_data and attn_data['spatial_probs'] is not None:
                # Use spatial probabilities as proxy for attention
                spatial_probs = attn_data['spatial_probs']
                if isinstance(spatial_probs, np.ndarray) and spatial_probs.ndim == 3:
                    saliency = np.max(spatial_probs, axis=0)
            elif 'saliency_map' in attn_data:
                sal_map = attn_data['saliency_map']
                if isinstance(sal_map, np.ndarray):
                    saliency = np.mean(sal_map, axis=0) if sal_map.ndim == 3 else sal_map
            
            if saliency is None:
                continue
            
            # Create fresh figure for each frame
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)  # Lower DPI for faster rendering
            ax.axis('off')
            
            saliency_up = upsample_attention(saliency, image.shape[:2])
            overlay = create_heatmap_overlay(image, saliency_up, self.colormap, self.alpha)
            
            ax.imshow(overlay)
            
            # Add title with timestep info and prediction change indicator
            timestep = attn_data.get('timestep', i)
            pred_class = attn_data.get('predicted_class', -1)
            
            # Check if prediction changed from previous frame
            prediction_changed = False
            if i > 0 and pred_class != -1:
                prev_pred = attention_sequence[i-1].get('predicted_class', -1)
                prediction_changed = (pred_class != prev_pred) and (prev_pred != -1)
            
            if pred_class != -1:
                class_name = self.class_names.get(str(pred_class), f"Class {pred_class}")
                title = f'Attention Evolution - Step {i+1}/{num_steps}\n'
                title += f'Timestep: {timestep} | Predicted: {class_name}'
                if prediction_changed:
                    title += ' [CHANGED]'
                ax.set_title(title, fontsize=12, fontweight='bold',
                           color='red' if prediction_changed else 'black')
            else:
                ax.set_title(f'Attention Evolution - Step {i+1}/{num_steps}\nTimestep: {timestep}', 
                           fontsize=12, fontweight='bold')
            
            # Adjust layout - no tight_layout needed for single subplot
            fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
            
            # Capture frame - ensure figure is fully rendered
            fig.canvas.draw()
            
            # Get the rendered image from buffer
            buf = fig.canvas.buffer_rgba()
            frame_rgba = np.asarray(buf)
            
            # Convert RGBA to RGB
            frame_rgb = frame_rgba[:, :, :3].copy()
            frames.append(frame_rgb)
            
            plt.close(fig)  # Close immediately
        
        # Ensure we have frames
        if len(frames) == 0:
            raise ValueError("No frames generated for stacked visualization")
        
        layout = (5, 2)
        max_panels = layout[0] * layout[1]
        frame_indices = select_key_frames(list(range(len(frames))), max_panels)
        frame_indices = sorted(frame_indices)
        selected_frames = [frames[i] for i in frame_indices]
        
        frame_titles = []
        for idx in frame_indices:
            attn_data = attention_sequence[idx]
            timestep = attn_data.get('timestep', idx)
            pred_class = attn_data.get('predicted_class', -1)
            if pred_class != -1:
                class_name = self.class_names.get(str(pred_class), f"Class {pred_class}")
                frame_titles.append(f"t={timestep}, {class_name}")
            else:
                frame_titles.append(f"t={timestep}")
        
        if save_path is None:
            save_path = 'attention_evolution.png'
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        save_frame_grid_as_image(
            selected_frames,
            save_path,
            layout=layout,
            titles=frame_titles,
            suptitle='Attention Evolution Through Diffusion Timesteps',
            dpi=self.dpi,
            frame_size=(3.5, 3.5),
            spacing=0.4
        )
        print(f"[AttentionVisualizer] Stacked frames saved to {save_path}")
        
        if self.save_pdf:
            pdf_path = save_path.with_suffix('.pdf')
            try:
                save_animation_frames_as_pdf(
                    selected_frames,
                    pdf_path,
                    layout=layout,
                    titles=frame_titles,
                    suptitle='Attention Evolution Through Diffusion Timesteps',
                    dpi=self.dpi,
                    frame_size=(3.5, 3.5),
                    spacing=0.4
                )
                print(f"[AttentionVisualizer] PDF with frames saved to {pdf_path}")
            except Exception as e:
                print(f"[AttentionVisualizer] Warning: Failed to save PDF: {e}")
        
        return save_path


"""
Usage Example:

from xai.visualizers.attention_vis import AttentionVisualizer
from xai.explainers.attention_explainer import AttentionExplainer

# Create visualizer
config = {
    'colormap': 'jet',
    'overlay_alpha': 0.5,
    'figure_size': [12, 8],
    'image_dpi': 300,
    'class_names': {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
}
visualizer = AttentionVisualizer(config)

# Generate explanation
explainer = AttentionExplainer(model, device, config)
explanation = explainer.explain(image_tensor, label=2)

# Create visualizations
vis1 = visualizer.visualize_attention(
    image, explanation,
    save_path='outputs/images/attention_overview.png'
)

vis2 = visualizer.visualize_all_class_saliency(
    image, explanation['saliency_map'],
    save_path='outputs/images/class_saliency.png'
)

vis3 = visualizer.visualize_patch_attention_bars(
    explanation['patch_attention'],
    save_path='outputs/images/patch_bars.png'
)

vis4 = visualizer.visualize_global_local_comparison(
    image, explanation,
    save_path='outputs/images/pathway_comparison.png'
)

Future Enhancements:
- Animation showing attention evolution during training
- Comparison between correct and incorrect predictions
- Statistical analysis of attention patterns across dataset
- Export to formats suitable for medical imaging software
"""

