"""
Image Utilities Module

This module provides common image processing functions used across the XAI pipeline.
All image loading, preprocessing, and visualization operations are centralized here.

Purpose:
- Load and preprocess images for model input
- Create attention heatmap overlays
- Draw bounding boxes for patches
- Upsample attention maps to match image resolution
- Save visualizations with consistent formatting

Future Extensions:
- Support for video/3D medical images
- Advanced color mapping techniques
- Automatic quality enhancement for visualizations
- Export to different formats (SVG, PDF)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_image(image_path: Union[str, Path],
               target_size: Optional[Tuple[int, int]] = None,
               return_original: bool = False) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        target_size: If provided, resize image to (width, height)
        return_original: If True, return both original and resized versions
    
    Returns:
        PIL Image, or tuple of (original_image, resized_image) if return_original=True
    
    Note:
        Uses PIL for consistency with torchvision transforms
        
    Future:
        - Add support for different color spaces
        - Handle DICOM medical images
        - Support for image sequences
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original = image.copy()
    
    # Resize if needed
    if target_size is not None:
        image = image.resize(target_size, Image.BILINEAR)
    
    if return_original:
        return original, image
    return image


def preprocess_for_model(image: Union[Image.Image, np.ndarray],
                         normalize_mean: List[float] = [0.485, 0.456, 0.406],
                         normalize_std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image or numpy array (H, W, C)
        normalize_mean: Mean values for normalization (ImageNet default)
        normalize_std: Std values for normalization (ImageNet default)
    
    Returns:
        Preprocessed tensor of shape (1, C, H, W) ready for model input
    
    Processing steps:
        1. Convert to numpy array if PIL Image
        2. Convert to float32 and scale to [0, 1]
        3. Normalize using mean and std
        4. Convert to torch tensor (C, H, W)
        5. Add batch dimension (1, C, H, W)
        
    Future:
        - Support for different normalization schemes
        - Automatic detection of normalization from model config
        - Data augmentation options for robustness testing
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure float32 and [0, 1] range
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(normalize_mean).reshape(1, 1, 3)
    std = np.array(normalize_std).reshape(1, 1, 3)
    image = (image - mean) / std
    
    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.float()


def upsample_attention(attention_map: Union[np.ndarray, torch.Tensor],
                      target_size: Tuple[int, int],
                      mode: str = 'bilinear') -> np.ndarray:
    """
    Upsample attention map to target image size.
    
    Args:
        attention_map: Attention map of shape (H, W) or (C, H, W)
        target_size: Target (height, width)
        mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
    
    Returns:
        Upsampled attention map as numpy array
    
    Note:
        Preserves the attention value distribution while increasing resolution.
        Uses bilinear interpolation by default for smooth gradients.
        
    Future:
        - Edge-aware upsampling to preserve boundaries
        - Super-resolution for better quality
        - Guided upsampling using original image
    """
    # Convert to tensor if numpy
    if isinstance(attention_map, np.ndarray):
        attention_tensor = torch.from_numpy(attention_map).float()
    else:
        attention_tensor = attention_map.float()
    
    # Add batch and channel dims if needed
    if attention_tensor.ndim == 2:  # (H, W)
        attention_tensor = attention_tensor.unsqueeze(0).unsqueeze(0)
    elif attention_tensor.ndim == 3:  # (C, H, W)
        attention_tensor = attention_tensor.unsqueeze(0)
    
    # Upsample
    upsampled = F.interpolate(
        attention_tensor,
        size=target_size,
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )
    
    # Remove batch dim and convert to numpy
    result = upsampled.squeeze(0).numpy()
    
    # If single channel, remove channel dim too
    if result.shape[0] == 1:
        result = result.squeeze(0)
    
    return result


def create_heatmap_overlay(image: Union[Image.Image, np.ndarray],
                           attention_map: np.ndarray,
                           colormap: str = 'jet',
                           alpha: float = 0.5,
                           normalize: bool = True) -> np.ndarray:
    """
    Create an attention heatmap overlay on the original image.
    
    Args:
        image: Original image (PIL or numpy array)
        attention_map: Attention map of same spatial size as image (H, W)
        colormap: Matplotlib colormap name ('jet', 'viridis', 'hot', etc.)
        alpha: Opacity of overlay (0=transparent, 1=opaque)
        normalize: Whether to normalize attention to [0, 1] range
    
    Returns:
        Blended image as numpy array (H, W, 3) in [0, 255] range
    
    Visualization Strategy:
        - High attention regions are highlighted with bright colors
        - Low attention regions remain closer to original image
        - Alpha blending preserves original image details
        
    Future:
        - Threshold-based highlighting (only show top-k%)
        - Smooth gradients for better aesthetics  
        - Support for multi-class attention (different colors per class)
    """
    # Convert image to numpy if PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB (H, W, 3)
    if image.ndim == 2:
        # Grayscale -> RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        # Check if it's RGBA (4 channels) and convert to RGB
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            # Single channel -> RGB
            image = np.repeat(image, 3, axis=2)
        # If already 3 channels, keep as is
    
    # Ensure uint8 range [0, 255]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        # Already uint8, but ensure it's in valid range
        image = np.clip(image, 0, 255)
    
    # Normalize attention if requested
    if normalize:
        attention_min = attention_map.min()
        attention_max = attention_map.max()
        if attention_max > attention_min:
            attention_map = (attention_map - attention_min) / (attention_max - attention_min)
        else:
            attention_map = np.zeros_like(attention_map)
    
    # Ensure attention is 2D
    if attention_map.ndim == 3:
        # If it's (1, H, W), squeeze the first dimension
        if attention_map.shape[0] == 1:
            attention_map = attention_map.squeeze(0)
        else:
            # If it's (H, W, 1) or similar, squeeze the last dimension
            attention_map = attention_map.squeeze()
    elif attention_map.ndim > 2:
        # Multiple dimensions, take the last 2
        attention_map = attention_map.reshape(-1, attention_map.shape[-2], attention_map.shape[-1])
        if attention_map.shape[0] == 1:
            attention_map = attention_map[0]
        else:
            # Average if multiple
            attention_map = attention_map.mean(axis=0)
    
    # Ensure attention_map is 2D now
    assert attention_map.ndim == 2, f"Expected 2D attention map, got shape {attention_map.shape}"
    
    # Resize attention to match image if needed
    if attention_map.shape != image.shape[:2]:
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    # Colormap returns (H, W, 4) for RGBA, we need RGB
    colored_attention = cmap(attention_map)  # Shape: (H, W, 4)
    
    # Extract RGB channels (drop alpha)
    if colored_attention.shape[2] == 4:
        colored_attention = colored_attention[:, :, :3]
    elif colored_attention.shape[2] == 3:
        pass  # Already RGB
    else:
        raise ValueError(f"Unexpected colormap output shape: {colored_attention.shape}")
    
    # Convert to uint8 [0, 255]
    colored_attention = (colored_attention * 255).astype(np.uint8)
    
    # Ensure both arrays have exactly the same shape and dtype
    # Image should be (H, W, 3), colored_attention should be (H, W, 3)
    if image.shape != colored_attention.shape:
        # Final safety check - resize colored_attention to match image exactly
        colored_attention = cv2.resize(
            colored_attention, 
            (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
    
    # Ensure same dtype
    if image.dtype != colored_attention.dtype:
        colored_attention = colored_attention.astype(image.dtype)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, colored_attention, alpha, 0)
    
    return overlay


def draw_bounding_boxes(image: Union[Image.Image, np.ndarray],
                       boxes: Union[List[Tuple], np.ndarray],
                       labels: Optional[List[str]] = None,
                       colors: Optional[List[Tuple]] = None,
                       thickness: int = 3,
                       font_scale: float = 0.8) -> np.ndarray:
    """
    Draw bounding boxes on image (for visualizing patch locations).
    
    Args:
        image: Original image
        boxes: List of boxes, each as (x, y, width, height) or (x1, y1, x2, y2)
        labels: Optional labels for each box
        colors: Optional colors for each box (RGB tuples)
        thickness: Line thickness
        font_scale: Font size for labels
    
    Returns:
        Image with bounding boxes drawn
    
    Use Cases:
        - Visualize attention patch locations
        - Show regions of interest
        - Display detected objects
        
    Future:
        - Filled vs outline boxes
        - Confidence-based opacity
        - Interactive box selection
    """
    # Convert image to numpy if PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Make a copy to avoid modifying original
    image = image.copy()
    
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Default colors if not provided
    if colors is None:
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
    
    # Draw each box
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        
        # Handle different box formats
        if len(box) == 4:
            x, y, w, h = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
        else:
            x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if labels is not None and i < len(labels):
            label = str(labels[i])
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2
            )
            
            # Draw background rectangle
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1  # Filled
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                2
            )
    
    return image


def save_visualization(image: np.ndarray,
                      save_path: Union[str, Path],
                      dpi: int = 300,
                      format: str = 'png') -> None:
    """
    Save visualization with consistent formatting.
    
    Args:
        image: Image to save (numpy array)
        save_path: Path to save the image
        dpi: Dots per inch for saved image
        format: Image format ('png', 'jpg', 'pdf', 'svg')
    
    Note:
        Creates parent directories if they don't exist.
        Uses high DPI for publication-quality figures.
        
    Future:
        - Automatic watermarking
        - Metadata embedding (model version, timestamp, etc.)
        - Compression options for large images
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Save using PIL for better format support
    pil_image = Image.fromarray(image)
    pil_image.save(save_path, format=format.upper(), dpi=(dpi, dpi))


def create_multi_panel_figure(images: List[np.ndarray],
                              titles: List[str],
                              suptitle: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 5),
                              cmap: Optional[str] = None) -> np.ndarray:
    """
    Create a multi-panel figure showing multiple images side by side.
    
    Args:
        images: List of images to display
        titles: Title for each image
        suptitle: Overall figure title
        figsize: Figure size (width, height)
        cmap: Colormap for grayscale images
    
    Returns:
        The figure as a numpy array (for saving)
    
    Use Cases:
        - Compare original vs attention overlay
        - Show progression through diffusion timesteps
        - Display multiple attention maps
        
    Future:
        - Customizable layouts (not just horizontal)
        - Zoom-in insets for regions of interest
        - Automatic aspect ratio handling
    """
    n_images = len(images)
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img.squeeze(), cmap=cmap if cmap else 'gray')
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image_array = np.asarray(buf)[:, :, :3]

    plt.close(fig)
    
    return image_array


def figure_to_array(fig) -> np.ndarray:
    """
    Convert matplotlib figure to numpy array (RGB).
    
    Args:
        fig: Matplotlib figure to convert
    
    Returns:
        RGB numpy array of shape (H, W, 3) with dtype uint8
    
    Note:
        This function renders the figure and converts it to an RGB array.
        Useful for saving visualizations or embedding in reports.
    """
    # Ensure figure is fully rendered
    fig.canvas.draw()
    
    # Get RGBA buffer
    buf = fig.canvas.buffer_rgba()
    frame_rgba = np.asarray(buf)
    
    # Convert RGBA to RGB
    frame_rgb = frame_rgba[:, :, :3].copy()
    
    # Convert to uint8 if needed
    if frame_rgb.dtype != np.uint8:
        if frame_rgb.max() <= 1.0:
            frame_rgb = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    
    return frame_rgb


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to uint8 [0, 255] range.
    
    Args:
        array: Input array of any dtype
    
    Returns:
        Uint8 array
        
    Handles different input ranges gracefully.
    """
    if array.dtype == np.uint8:
        return array
    
    # Find min and max
    arr_min = array.min()
    arr_max = array.max()
    
    # Normalize to [0, 1]
    if arr_max > arr_min:
        normalized = (array - arr_min) / (arr_max - arr_min)
    else:
        normalized = np.zeros_like(array, dtype=np.float32)
    
    # Scale to [0, 255]
    return (normalized * 255).astype(np.uint8)


"""
Usage Examples:

# Load and preprocess image
image = load_image('path/to/image.png', target_size=(512, 512))
tensor = preprocess_for_model(image)

# Create attention overlay
attention = model.get_attention(tensor)  # Shape: (16, 16)
attention_upsampled = upsample_attention(attention, (512, 512))
overlay = create_heatmap_overlay(image, attention_upsampled, alpha=0.5)

# Draw patch bounding boxes
boxes = [(100, 100, 73, 73), (200, 200, 73, 73)]  # (x, y, w, h)
labels = ['Patch 1 (0.85)', 'Patch 2 (0.72)']
img_with_boxes = draw_bounding_boxes(overlay, boxes, labels)

# Save
save_visualization(img_with_boxes, 'output/attention_overlay.png')

# Create comparison figure
original = np.array(image)
figures = create_multi_panel_figure(
    [original, attention_upsampled, overlay],
    ['Original', 'Attention', 'Overlay'],
    suptitle='Attention Visualization'
)
save_visualization(figures, 'output/comparison.png')

Future Enhancements:
1. Video support for temporal attention
2. 3D visualization for volumetric medical images
3. Interactive web-based visualizations
4. Automatic layout optimization for different screen sizes
5. Export to LaTeX/TikZ for publications
"""

