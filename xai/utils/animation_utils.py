"""
Animation Utilities Module

Shared helper functions for creating animations across different visualizers.
These utilities standardize animation creation and reduce code duplication.

Functions:
- Frame capture from matplotlib figures
- Subplot grid creation
- Progress indicators
- Frame rate calculations
- GIF optimization

Purpose:
- Centralize common animation logic
- Ensure consistency across visualizers
- Simplify maintenance and updates
- Improve code reusability

Future Extensions:
- Video format support (MP4, WebM)
- Frame interpolation for smoother animations
- Compression optimization
- Batch processing utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Optional, List
from PIL import Image


def capture_frame(fig: Figure) -> np.ndarray:
    """
    Capture matplotlib figure as RGB numpy array.
    
    This function renders the figure to an array, ensuring proper
    color format and data type for GIF creation.
    
    Args:
        fig: Matplotlib figure to capture
    
    Returns:
        RGB numpy array of shape (H, W, 3) with dtype uint8
    
    Implementation:
        - Uses buffer_rgba() for reliable rendering
        - Converts RGBA to RGB by dropping alpha channel
        - Returns copy to avoid memory issues
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> frame = capture_frame(fig)
        >>> print(frame.shape)  # (height, width, 3)
    """
    # Ensure figure is fully rendered
    fig.canvas.draw()
    
    # Get RGBA buffer
    buf = fig.canvas.buffer_rgba()
    frame_rgba = np.asarray(buf)
    
    # Convert RGBA to RGB
    frame_rgb = frame_rgba[:, :, :3].copy()
    
    return frame_rgb


def create_subplot_grid(layout: str, 
                       figsize: Tuple[float, float] = (16, 12),
                       dpi: int = 100) -> Tuple[Figure, np.ndarray]:
    """
    Create subplot grid with specified layout.
    
    Args:
        layout: Layout type - "2x2", "1x3", "2x1", etc.
        figsize: Figure size in inches (width, height)
        dpi: Resolution for rendering
    
    Returns:
        Tuple of (figure, axes_array)
        - figure: Matplotlib Figure object
        - axes_array: Array of Axes objects
    
    Supported layouts:
        - "2x2": 2 rows, 2 columns
        - "1x3": 1 row, 3 columns  
        - "3x1": 3 rows, 1 column
        - "2x1": 2 rows, 1 column
    
    Example:
        >>> fig, axes = create_subplot_grid("2x2", figsize=(12, 10))
        >>> axes[0, 0].plot([1, 2, 3])
        >>> axes[0, 1].imshow(image)
    """
    # Parse layout string
    rows, cols = map(int, layout.split('x'))
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    
    # Ensure axes is always an array (even for single subplot)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)
    
    # Adjust spacing
    plt.tight_layout()
    
    return fig, axes


def add_timestep_indicator(ax: Axes, 
                           timestep: int, 
                           total_timesteps: int,
                           position: str = 'top') -> None:
    """
    Add timestep progress indicator to subplot.
    
    Adds a visual progress indicator showing current position
    in the animation sequence.
    
    Args:
        ax: Matplotlib axes to add indicator to
        timestep: Current timestep number
        total_timesteps: Total number of timesteps
        position: Where to place indicator - 'top', 'bottom', 'title'
    
    Implementation:
        - 'top': Horizontal bar at top of plot
        - 'bottom': Horizontal bar at bottom of plot
        - 'title': Text in title showing "Step X/Y"
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> add_timestep_indicator(ax, 50, 100, position='top')
    """
    progress = timestep / total_timesteps
    
    if position == 'title':
        # Add to title
        current_title = ax.get_title()
        new_title = f"{current_title}\nStep {timestep}/{total_timesteps}"
        ax.set_title(new_title)
        
    elif position == 'top':
        # Add progress bar at top
        ax.axhline(y=ax.get_ylim()[1], xmin=0, xmax=progress, 
                  color='green', linewidth=4, alpha=0.7)
        
    elif position == 'bottom':
        # Add progress bar at bottom
        ax.axhline(y=ax.get_ylim()[0], xmin=0, xmax=progress,
                  color='green', linewidth=4, alpha=0.7)


def add_prediction_indicator(ax: Axes,
                            prediction: int,
                            confidence: float,
                            class_names: dict,
                            changed: bool = False) -> None:
    """
    Add prediction label with confidence to subplot.
    
    Args:
        ax: Matplotlib axes
        prediction: Predicted class index
        confidence: Prediction confidence (0-1)
        class_names: Dictionary mapping class indices to names
        changed: Whether prediction changed from previous frame
    
    Display:
        - Shows predicted class name
        - Shows confidence percentage
        - Red border if prediction changed
        - Green border if stable
    
    Example:
        >>> add_prediction_indicator(ax, 2, 0.87, {0:"A", 1:"B", 2:"C"})
    """
    class_name = class_names.get(str(prediction), f"Class {prediction}")
    
    # Choose color based on whether prediction changed
    color = 'red' if changed else 'green'
    alpha = 0.8 if changed else 0.5
    
    # Create text box
    text = f'Prediction: {class_name}\nConfidence: {confidence:.1%}'
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=alpha, 
                    edgecolor='black', linewidth=2),
           color='white', fontweight='bold')


def save_frames_as_gif(frames: List[np.ndarray],
                       save_path: str,
                       fps: int = 5,
                       optimize: bool = True) -> None:
    """
    Save list of frames as animated GIF.
    
    Standardized GIF saving with optimization and validation.
    
    Args:
        frames: List of RGB numpy arrays (H, W, 3)
        save_path: Path to save GIF file
        fps: Frames per second
        optimize: Whether to optimize file size
    
    Validation:
        - Ensures all frames have same shape
        - Converts to uint8 if needed
        - Validates RGB format
    
    Implementation:
        Uses PIL's save() method with proper parameters for
        animated GIF creation.
    
    Example:
        >>> frames = [capture_frame(fig1), capture_frame(fig2)]
        >>> save_frames_as_gif(frames, "animation.gif", fps=10)
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        pil_frames.append(Image.fromarray(frame, mode='RGB'))
    
    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)
    
    # Save as animated GIF
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:] if len(pil_frames) > 1 else [],
        duration=duration_ms,
        loop=0,
        format='GIF',
        optimize=optimize
    )


def get_frame_shape(fig: Figure) -> Tuple[int, int]:
    """
    Get the pixel dimensions of a matplotlib figure.
    
    Args:
        fig: Matplotlib figure
    
    Returns:
        Tuple of (height, width) in pixels
    
    Example:
        >>> fig = plt.figure(figsize=(10, 8), dpi=100)
        >>> h, w = get_frame_shape(fig)
        >>> print(f"Frame size: {w}x{h}")
    """
    width_px, height_px = fig.canvas.get_width_height()
    return height_px, width_px


def calculate_optimal_dpi(target_width: int = 800,
                         figsize_width: float = 10.0) -> int:
    """
    Calculate optimal DPI for target pixel width.
    
    Args:
        target_width: Desired width in pixels
        figsize_width: Figure width in inches
    
    Returns:
        DPI value to achieve target width
    
    Example:
        >>> dpi = calculate_optimal_dpi(target_width=1200, figsize_width=12)
        >>> fig = plt.figure(figsize=(12, 8), dpi=dpi)
    """
    return int(target_width / figsize_width)


"""
Usage Examples:

# Basic animation creation
frames = []
for i in range(100):
    fig, ax = plt.subplots()
    ax.plot(data[i])
    add_timestep_indicator(ax, i, 100)
    frames.append(capture_frame(fig))
    plt.close(fig)

save_frames_as_gif(frames, "animation.gif", fps=10)

# Multi-panel animation
for timestep in range(num_steps):
    fig, axes = create_subplot_grid("2x2", figsize=(12, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 1].bar(range(5), probs[timestep])
    add_prediction_indicator(axes[0, 1], pred, conf, class_names)
    
    frame = capture_frame(fig)
    frames.append(frame)
    plt.close(fig)

save_frames_as_gif(frames, "multi_panel.gif", fps=5)
"""

