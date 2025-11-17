"""
Visualization modules for creating XAI outputs.

This package contains visualizers that convert explanations into
human-interpretable formats:
- attention_vis: Heatmap overlays and patch highlighting
- trajectory_vis: Diffusion process visualization
- report_generator: HTML report generation

Future extensions:
- comparison_visualizer: Side-by-side method comparisons
- interactive_visualizer: Web-based interactive exploration
"""

from .attention_vis import AttentionVisualizer
from .trajectory_vis import TrajectoryVisualizer
from .report_generator import ReportGenerator

__all__ = ['AttentionVisualizer', 'TrajectoryVisualizer', 'ReportGenerator']

