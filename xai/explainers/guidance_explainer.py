"""
Guidance Map Explainer Module

Extracts and analyzes the dense 2D guidance map M that interpolates between
global and local priors from the DCG auxiliary model.

Features:
- Extracts global prior (y_g) from full-image pathway
- Extracts local prior (y_l) from patch-based pathway
- Computes distance matrix for spatial interpolation
- Generates dense guidance map M of shape [C, N_p, N_p]
- Tracks interpolation weights (global vs local contribution)
- Provides per-class guidance distributions

Purpose:
- Understand how global and local priors are combined
- Analyze spatial interpolation mechanisms
- Visualize guidance signal distribution across classes
- Support research on multi-scale feature fusion

Future Extensions:
- Guidance map evolution through diffusion timesteps
- Class-specific guidance analysis
- Interpolation weight optimization
- Multi-scale guidance visualization
- Guidance map clustering and pattern analysis
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class GuidanceMapExplainer(BaseExplainer):
    """
    Extract dense guidance maps from the auxiliary DCG model.
    
    The guidance map M interpolates between global and local priors:
    - Global prior (y_g): Full-image prediction
    - Local prior (y_l): Patch-based predictions
    - Distance-based interpolation: Closer to diagonal = more global
    
    Attributes:
        model: CoolSystem containing aux_model
        aux_model: The DCG auxiliary classifier
        config: Configuration dictionary
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the guidance map explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with settings
        """
        super().__init__(model, device, config)
        
        self.aux_model = model.aux_model
        self.aux_model.eval()
        
        self.num_classes = config.get('num_classes', 5)
        self.model_config = model.params
        
        print(f"[GuidanceMapExplainer] Initialized")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate guidance map explanation.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - 'explanation_type': 'guidance_map'
            - 'prediction': Predicted class
            - 'confidence': Confidence score
            - 'guidance_map': Dense guidance map [C, N_p, N_p]
            - 'global_prior': Global prior probabilities [C]
            - 'local_prior': Local prior probabilities [C]
            - 'distance_matrix': Distance matrix [N_p, N_p]
            - 'interpolation_weights': Weight matrices for interpolation
        """
        image = self._preprocess_image(image)
        
        with torch.no_grad():
            (y_fusion, y_global, y_local,
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        bz = image.shape[0]
        nc = self.num_classes
        
        # Get patch count from saliency map dimensions
        _, _, H, W = saliency_map.shape
        np_patches = int(np.sqrt(H * W))
        
        # Create guidance map
        guidance_map = self._create_guidance_map(y_global, y_local, bz, nc, np_patches)
        
        # Compute distance matrix and weights
        distance_matrix, weight_g, weight_l = self._compute_interpolation_weights(np_patches)
        
        # Get predictions
        probs_global = F.softmax(y_global, dim=1)[0]
        probs_local = F.softmax(y_local, dim=1)[0]
        probs_fusion = F.softmax(y_fusion, dim=1)[0]
        
        prediction = int(torch.argmax(probs_fusion, dim=0).item())
        confidence = float(probs_fusion[prediction].item())
        
        explanation = self.get_explanation_dict(
            explanation_type='guidance_map',
            prediction=prediction,
            confidence=confidence,
            
            # Guidance map data
            guidance_map=guidance_map[0].detach().cpu().numpy(),  # [C, N_p, N_p]
            global_prior=probs_global.detach().cpu().numpy(),
            local_prior=probs_local.detach().cpu().numpy(),
            fusion_prior=probs_fusion.detach().cpu().numpy(),
            
            # Interpolation components
            distance_matrix=distance_matrix.detach().cpu().numpy(),
            global_weight=weight_g.detach().cpu().numpy(),
            local_weight=weight_l.detach().cpu().numpy(),
            
            # Raw logits (before softmax)
            global_logits=y_global[0].detach().cpu().numpy(),
            local_logits=y_local[0].detach().cpu().numpy(),
            fusion_logits=y_fusion[0].detach().cpu().numpy(),
            
            # Patch information
            num_patches=np_patches,
            patch_attention=patch_attns[0].detach().cpu().numpy() if patch_attns is not None else None,
            
            # Ground truth
            ground_truth=label if label is not None else -1
        )
        
        return explanation
    
    def _create_guidance_map(self, y0_g: torch.Tensor, y0_l: torch.Tensor,
                            bz: int, nc: int, np_patches: int) -> torch.Tensor:
        """
        Create dense guidance map by interpolating global and local priors.
        
        This replicates the guided_prob_map method from CoolSystem.
        
        Args:
            y0_g: Global prior logits [B, C]
            y0_l: Local prior logits [B, C]
            bz: Batch size
            nc: Number of classes
            np_patches: Number of patches (spatial dimension)
            
        Returns:
            Guidance map [B, C, N_p, N_p]
        """
        device = y0_g.device
        
        # Compute interpolation weights
        distance_matrix, weight_g, weight_l = self._compute_interpolation_weights(np_patches)
        
        # Convert logits to probabilities for interpolation
        probs_g = F.softmax(y0_g, dim=1)  # [B, C]
        probs_l = F.softmax(y0_l, dim=1)  # [B, C]
        
        # Interpolate: weight_l * local + weight_g * global
        interpolated = (
            weight_l.unsqueeze(0).unsqueeze(0).to(device) * probs_l.unsqueeze(-1).unsqueeze(-1) +
            weight_g.unsqueeze(0).unsqueeze(0).to(device) * probs_g.unsqueeze(-1).unsqueeze(-1)
        )  # [B, C, N_p, N_p]
        
        # Set diagonal to global values
        diag_indices = torch.arange(np_patches, device=device)
        guidance_map = interpolated.clone()
        
        for i in range(bz):
            for j in range(nc):
                guidance_map[i, j, diag_indices, diag_indices] = probs_g[i, j]
                # Set corners to local values
                guidance_map[i, j, np_patches-1, 0] = probs_l[i, j]
                guidance_map[i, j, 0, np_patches-1] = probs_l[i, j]
        
        return guidance_map
    
    def _compute_interpolation_weights(self, np_patches: int) -> tuple:
        """
        Compute distance-based interpolation weights.
        
        Args:
            np_patches: Number of patches (spatial dimension)
            
        Returns:
            Tuple of (distance_matrix, weight_global, weight_local)
        """
        device = self.device
        
        # Distance to diagonal: |i - j|
        distance_matrix = torch.tensor(
            [[abs(i - j) for j in range(np_patches)] for i in range(np_patches)],
            device=device,
            dtype=torch.float32
        )
        
        # Normalize to [0, 1]
        max_distance = np_patches - 1
        normalized_distance = distance_matrix / max_distance if max_distance > 0 else distance_matrix
        
        # Weight for global (decreases with distance from diagonal)
        weight_g = 1 - normalized_distance
        
        # Weight for local (increases with distance from diagonal)
        weight_l = normalized_distance
        
        return distance_matrix, weight_g, weight_l

