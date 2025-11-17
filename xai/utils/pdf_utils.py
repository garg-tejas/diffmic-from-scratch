"""
PDF Export Utilities for Research Paper Figures

This module provides utilities for exporting visualizations as publication-quality PDFs.
It handles both static figures and animation frames (as grids or multi-page PDFs).

Features:
- High-quality PDF export with proper fonts and layouts
- Animation frame export (grid layout or multi-page)
- Key frame selection for animations
- Publication-ready formatting
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple, Union, Dict, Any
from PIL import Image
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def save_figure_as_pdf(fig: plt.Figure,
                       save_path: Union[str, Path],
                       dpi: int = 300,
                       bbox_inches: str = 'tight',
                       pad_inches: float = 0.1) -> None:
    """
    Save a matplotlib figure as a high-quality PDF.
    
    Args:
        fig: Matplotlib figure to save
        save_path: Path to save PDF
        dpi: Resolution (for rasterized elements)
        bbox_inches: Bounding box mode ('tight', 'standard', etc.)
        pad_inches: Padding around figure in inches
        
    Note:
        Creates parent directories if they don't exist.
        Uses vector format for best quality in publications.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        save_path,
        format='pdf',
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        facecolor='white',
        edgecolor='none',
        transparent=False
    )


def save_animation_frames_as_pdf(frames: List[np.ndarray],
                                  save_path: Union[str, Path],
                                  layout: Tuple[int, int] = (2, 5),
                                  titles: Optional[List[str]] = None,
                                  suptitle: Optional[str] = None,
                                  dpi: int = 300,
                                  frame_size: Tuple[float, float] = (3, 3),
                                  spacing: float = 0.3) -> None:
    """
    Save animation frames as a PDF with grid layout.
    
    Args:
        frames: List of frame images (numpy arrays)
        save_path: Path to save PDF
        layout: Grid layout (rows, cols)
        titles: Optional titles for each frame
        suptitle: Optional main title for the figure
        dpi: Resolution
        frame_size: Size of each frame in inches (width, height)
        spacing: Spacing between frames in inches
        
    Note:
        If there are more frames than grid slots, creates multiple pages.
        Selects key frames evenly distributed across the sequence.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_rows, n_cols = layout
    frames_per_page = n_rows * n_cols
    
    # Select key frames if there are too many
    if len(frames) > frames_per_page:
        # Select evenly distributed frames
        indices = np.linspace(0, len(frames) - 1, frames_per_page, dtype=int)
        selected_frames = [frames[i] for i in indices]
        selected_titles = [titles[i] if titles and i < len(titles) else f"Frame {i+1}" 
                          for i in indices] if titles else None
    else:
        selected_frames = frames
        selected_titles = titles if titles else [f"Frame {i+1}" for i in range(len(frames))]
    
    # Calculate figure size
    fig_width = n_cols * frame_size[0] + (n_cols + 1) * spacing
    fig_height = n_rows * frame_size[1] + (n_rows + 1) * spacing + (0.5 if suptitle else 0)
    
    # Create PDF with multiple pages if needed
    with PdfPages(save_path) as pdf:
        num_pages = (len(selected_frames) + frames_per_page - 1) // frames_per_page
        
        for page in range(num_pages):
            start_idx = page * frames_per_page
            end_idx = min(start_idx + frames_per_page, len(selected_frames))
            page_frames = selected_frames[start_idx:end_idx]
            page_titles = selected_titles[start_idx:end_idx] if selected_titles else None
            
            # Create figure for this page
            fig = plt.figure(figsize=(fig_width, fig_height))
            if suptitle:
                fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.0 - 0.5/fig_height)
            
            gs = GridSpec(n_rows, n_cols, figure=fig,
                         left=spacing/fig_width,
                         right=1 - spacing/fig_width,
                         top=1 - (spacing + (0.5 if suptitle else 0))/fig_height,
                         bottom=spacing/fig_height,
                         wspace=spacing/frame_size[0],
                         hspace=spacing/frame_size[1])
            
            # Plot frames
            for idx, frame in enumerate(page_frames):
                row = idx // n_cols
                col = idx % n_cols
                
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(frame, aspect='auto')
                ax.axis('off')
                
                if page_titles and idx < len(page_titles):
                    ax.set_title(page_titles[idx], fontsize=10, pad=5)
                else:
                    frame_num = start_idx + idx + 1
                    ax.set_title(f"Frame {frame_num}", fontsize=10, pad=5)
            
            # Hide unused subplots
            for idx in range(len(page_frames), frames_per_page):
                row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
            plt.close(fig)


def save_frame_grid_as_image(frames: List[np.ndarray],
                             save_path: Union[str, Path],
                             layout: Tuple[int, int] = (5, 2),
                             titles: Optional[List[str]] = None,
                             suptitle: Optional[str] = None,
                             dpi: int = 300,
                             frame_size: Tuple[float, float] = (3.5, 3.5),
                             spacing: float = 0.4) -> None:
    """
    Save animation frames as a single PNG image arranged in a grid.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_rows, n_cols = layout
    frames_per_grid = n_rows * n_cols
    
    if len(frames) > frames_per_grid:
        indices = np.linspace(0, len(frames) - 1, frames_per_grid, dtype=int)
        selected_frames = [frames[i] for i in indices]
        selected_titles = [titles[i] if titles and i < len(titles) else f"Frame {i+1}"
                           for i in indices] if titles else None
    else:
        selected_frames = frames
        selected_titles = titles if titles else [f"Frame {i+1}" for i in range(len(frames))]
    
    fig_width = n_cols * frame_size[0] + (n_cols + 1) * spacing
    fig_height = n_rows * frame_size[1] + (n_rows + 1) * spacing + (0.5 if suptitle else 0)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.0 - 0.5 / fig_height)
    
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  left=spacing / fig_width,
                  right=1 - spacing / fig_width,
                  top=1 - (spacing + (0.5 if suptitle else 0)) / fig_height,
                  bottom=spacing / fig_height,
                  wspace=spacing / frame_size[0],
                  hspace=spacing / frame_size[1])
    
    for idx, frame in enumerate(selected_frames):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(frame)
        ax.axis('off')
        if selected_titles and idx < len(selected_titles):
            ax.set_title(selected_titles[idx], fontsize=10, pad=5)
        else:
            frame_num = idx + 1
            ax.set_title(f"Frame {frame_num}", fontsize=10, pad=5)
    
    for idx in range(len(selected_frames), frames_per_grid):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_animation_frames_multi_page(frames: List[np.ndarray],
                                     save_path: Union[str, Path],
                                     titles: Optional[List[str]] = None,
                                     dpi: int = 300,
                                     frame_size: Tuple[float, float] = (6, 6)) -> None:
    """
    Save animation frames as a multi-page PDF (one frame per page).
    
    Args:
        frames: List of frame images (numpy arrays)
        save_path: Path to save PDF
        titles: Optional titles for each frame
        dpi: Resolution
        frame_size: Size of each frame in inches (width, height)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(save_path) as pdf:
        for idx, frame in enumerate(frames):
            fig, ax = plt.subplots(figsize=frame_size)
            ax.imshow(frame, aspect='auto')
            ax.axis('off')
            
            if titles and idx < len(titles):
                fig.suptitle(titles[idx], fontsize=12, fontweight='bold')
            else:
                fig.suptitle(f"Frame {idx + 1}", fontsize=12, fontweight='bold')
            
            pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
            plt.close(fig)


def select_key_frames(frames: List[np.ndarray],
                     num_frames: int = 10,
                     method: str = 'uniform') -> List[np.ndarray]:
    """
    Select key frames from an animation sequence.
    
    Args:
        frames: List of all frames
        num_frames: Number of key frames to select
        method: Selection method ('uniform', 'start_end', 'adaptive')
        
    Returns:
        List of selected key frames
    """
    if len(frames) <= num_frames:
        return frames
    
    if method == 'uniform':
        # Uniformly sample frames
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]
    
    elif method == 'start_end':
        # Emphasize start and end, sample middle uniformly
        start_frames = frames[:2]
        end_frames = frames[-2:]
        middle_frames = frames[2:-2]
        
        if len(middle_frames) > num_frames - 4:
            middle_indices = np.linspace(0, len(middle_frames) - 1, 
                                       num_frames - 4, dtype=int)
            selected_middle = [middle_frames[i] for i in middle_indices]
        else:
            selected_middle = middle_frames
        
        return start_frames + selected_middle + end_frames
    
    elif method == 'adaptive':
        # TODO: Implement adaptive selection based on frame differences
        # For now, use uniform
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]
    
    else:
        raise ValueError(f"Unknown selection method: {method}")


def create_publication_style():
    """
    Set matplotlib style for publication-quality figures.
    
    Returns:
        Dictionary of style parameters
    """
    return {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'axes.labelweight': 'normal',
        'axes.titleweight': 'bold',
    }


def apply_publication_style():
    """Apply publication style to matplotlib."""
    style = create_publication_style()
    plt.rcParams.update(style)

