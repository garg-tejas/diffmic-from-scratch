"""
t-SNE Visualizer Module

Generates t-SNE visualizations of diffusion embeddings across timesteps.
Shows how class separation improves as diffusion progresses.

Purpose:
- Fig 9: t-SNE progression across timesteps (t=0, 20, 50, 100)
- Visualize temporal decision-space evolution
- Demonstrate interpretability of diffusion classification

Usage:
    >>> visualizer = TSNEVisualizer(config)
    >>> visualizer.create_tsne_progression(
    ...     embeddings_dict={
    ...         't0': features_t0,  # [N, num_classes]
    ...         't20': features_t20,
    ...         't50': features_t50,
    ...         't100': features_t100
    ...     },
    ...     true_labels=labels,  # [N]
    ...     predicted_labels=preds,  # [N]
    ...     save_path='fig9_tsne_progression.png'
    ... )
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import seaborn as sns

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18


class TSNEVisualizer:
    """
    Visualize t-SNE embeddings of diffusion features across timesteps.
    
    Creates 4-panel progression showing how class separation improves
    as the diffusion model denoises from t=1000 to t=0.
    
    Attributes:
        config: Configuration dictionary with visualization settings
        class_names: Dictionary mapping class_id -> class_name
        colormap: Colormap for class coloring
        marker_correct: Marker style for correct predictions
        marker_incorrect: Marker style for incorrect predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize t-SNE visualizer.
        
        Args:
            config: Configuration dictionary with:
                - class_names: Dict[int, str] mapping class IDs to names
                - colormap: Colormap name (default: 'tab10')
                - figure_size: Figure size tuple (default: (16, 4))
                - image_dpi: DPI for saved figures (default: 300)
                - tsne_perplexity: t-SNE perplexity (default: 30)
                - tsne_learning_rate: t-SNE learning rate (default: 200)
                - tsne_n_iter: t-SNE iterations (default: 1000)
        """
        self.config = config
        self.class_names = config.get('class_names', {
            0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 
            3: 'Class 3', 4: 'Class 4'
        })
        self.colormap = config.get('colormap', 'tab10')
        self.figure_size = config.get('figure_size', (16, 4))
        self.image_dpi = config.get('image_dpi', 300)
        
        # t-SNE parameters
        self.tsne_perplexity = config.get('tsne_perplexity', 30)
        self.tsne_learning_rate = config.get('tsne_learning_rate', 200)
        self.tsne_n_iter = config.get('tsne_n_iter', 1000)
        
        # Marker styles
        self.marker_correct = config.get('marker_correct', 'o')
        self.marker_incorrect = config.get('marker_incorrect', 'X')
        
        print(f"[TSNEVisualizer] Initialized")
        print(f"  t-SNE perplexity: {self.tsne_perplexity}")
        print(f"  t-SNE learning rate: {self.tsne_learning_rate}")
        print(f"  t-SNE iterations: {self.tsne_n_iter}")
    
    def create_tsne_progression(self,
                               embeddings_dict: Dict[str, np.ndarray],
                               true_labels: np.ndarray,
                               predicted_labels: Optional[np.ndarray] = None,
                               save_path: Optional[Path] = None,
                               timestep_labels: Optional[list] = None) -> plt.Figure:
        """
        Create 4-panel t-SNE progression figure.
        
        Args:
            embeddings_dict: Dictionary with keys 't0', 't20', 't50', 't100'
                           Each value is [N, num_classes] array
            true_labels: True class labels [N]
            predicted_labels: Predicted labels [N] (optional, for marking incorrect)
            save_path: Path to save figure (optional)
            timestep_labels: List of timestep labels (default: ['t=0', 't=20', 't=50', 't=100'])
        
        Returns:
            matplotlib Figure object
        """
        # Default timestep labels
        if timestep_labels is None:
            timestep_labels = ['t=0', 't=20', 't=50', 't=100']
        
        # Expected keys
        expected_keys = ['t0', 't20', 't50', 't100']
        
        # Verify all timesteps are present
        for key in expected_keys:
            if key not in embeddings_dict:
                raise ValueError(f"Missing embedding key: {key}")
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=self.figure_size)
        fig.suptitle('t-SNE Embedding Evolution Across Diffusion Timesteps', 
                     fontsize=18, fontweight='bold')
        
        # Get unique classes for consistent coloring
        unique_classes = np.unique(true_labels)
        num_classes = len(unique_classes)
        
        # Create color map
        colors = plt.cm.get_cmap(self.colormap, num_classes)
        class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
        
        # Process each timestep
        for idx, (t_key, t_label) in enumerate(zip(expected_keys, timestep_labels)):
            ax = axes[idx]
            
            # Get embeddings for this timestep
            features = embeddings_dict[t_key]  # [N, num_classes]
            
            # Run t-SNE
            print(f"Running t-SNE for {t_key}...")
            tsne = TSNE(
                n_components=2,
                perplexity=self.tsne_perplexity,
                learning_rate=self.tsne_learning_rate,
                n_iter=self.tsne_n_iter,
                random_state=42,
                verbose=0
            )
            
            embeddings_2d = tsne.fit_transform(features)
            
            # Plot by class
            if predicted_labels is not None:
                # Mark correct vs incorrect predictions
                correct_mask = (true_labels == predicted_labels)
                incorrect_mask = ~correct_mask
                
                # Plot correct predictions
                for cls in unique_classes:
                    cls_mask = (true_labels == cls) & correct_mask
                    if np.any(cls_mask):
                        ax.scatter(
                            embeddings_2d[cls_mask, 0],
                            embeddings_2d[cls_mask, 1],
                            c=[class_to_color[cls]],
                            label=f'{self.class_names.get(cls, f"Class {cls}")} (correct)',
                            marker=self.marker_correct,
                            alpha=0.6,
                            s=30
                        )
                
                # Plot incorrect predictions
                if np.any(incorrect_mask):
                    ax.scatter(
                        embeddings_2d[incorrect_mask, 0],
                        embeddings_2d[incorrect_mask, 1],
                        c='red',
                        label='Incorrect',
                        marker=self.marker_incorrect,
                        alpha=0.8,
                        s=50,
                        edgecolors='black',
                        linewidths=1
                    )
            else:
                # Just plot by class
                for cls in unique_classes:
                    cls_mask = (true_labels == cls)
                    if np.any(cls_mask):
                        ax.scatter(
                            embeddings_2d[cls_mask, 0],
                            embeddings_2d[cls_mask, 1],
                            c=[class_to_color[cls]],
                            label=self.class_names.get(cls, f'Class {cls}'),
                            alpha=0.6,
                            s=30
                        )
            
            # Customize subplot
            ax.set_title(t_label, fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            if idx == 0:
                ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.image_dpi, bbox_inches='tight')
            print(f"Saved t-SNE progression to {save_path}")
        
        return fig
    
    def create_single_tsne(self,
                          features: np.ndarray,
                          labels: np.ndarray,
                          title: str = 't-SNE Embedding',
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create single t-SNE plot (for debugging or individual analysis).
        
        Args:
            features: Feature embeddings [N, num_classes]
            labels: Class labels [N]
            title: Plot title
            save_path: Path to save figure (optional)
        
        Returns:
            matplotlib Figure object
        """
        # Run t-SNE
        print(f"Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            perplexity=self.tsne_perplexity,
            learning_rate=self.tsne_learning_rate,
            n_iter=self.tsne_n_iter,
            random_state=42
        )
        
        embeddings_2d = tsne.fit_transform(features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique classes
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        
        # Create color map
        colors = plt.cm.get_cmap(self.colormap, num_classes)
        
        # Plot by class
        for i, cls in enumerate(unique_classes):
            cls_mask = (labels == cls)
            ax.scatter(
                embeddings_2d[cls_mask, 0],
                embeddings_2d[cls_mask, 1],
                c=[colors(i)],
                label=self.class_names.get(cls, f'Class {cls}'),
                alpha=0.6,
                s=30
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.image_dpi, bbox_inches='tight')
            print(f"Saved t-SNE plot to {save_path}")
        
        return fig
    
    def load_embeddings_from_npz(self, npz_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Load embeddings from saved .npz file.
        
        Args:
            npz_path: Path to .npz file created by test_model.py
        
        Returns:
            Tuple of (embeddings_dict, true_labels, predicted_labels)
            where embeddings_dict has keys: 't0', 't20', 't50', 't100'
        """
        data = np.load(npz_path)
        
        embeddings_dict = {
            't0': data['features'],
            't20': data['features'],  # Will be overwritten if multiple timesteps saved
            't50': data['features'],
            't100': data['features']
        }
        
        # Check if multiple timesteps are saved separately
        # (This depends on how test_model.py saves the data)
        # For now, assume single timestep per file
        # In practice, you'd load multiple files and combine
        
        true_labels = data['true_labels']
        predicted_labels = data.get('predicted_labels', None)
        
        return embeddings_dict, true_labels, predicted_labels

