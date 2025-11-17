"""
Feature Prior Explainer Module

Analyzes the image feature prior F that combines Transformer (global) and
CNN (local) features to condition the diffusion denoising process.

Features:
- Extracts raw features from Transformer encoder (full image)
- Extracts ROI features from CNN encoder (local patches)
- Analyzes learnable fusion weights Q
- Computes feature contribution scores
- Tracks feature magnitudes and norms
- Provides ROI importance rankings

Purpose:
- Understand how different feature sources contribute to predictions
- Analyze feature fusion mechanisms
- Identify which ROIs are most important
- Support feature engineering and model optimization
- Enable research on multi-modal feature fusion

Future Extensions:
- Feature prior evolution through diffusion timesteps
- Attention-weighted feature analysis
- Feature clustering and similarity analysis
- Counterfactual feature analysis
- Feature ablation studies
- Multi-scale feature pyramid visualization
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class FeaturePriorExplainer(BaseExplainer):
    """
    Analyze feature prior composition from image and ROI features.
    
    The feature prior F combines:
    - F_raw: Transformer encoder output (full image)
    - F_rois: CNN encoder outputs (local patches/ROIs)
    - Q: Learnable fusion weights
    
    Attributes:
        model: CoolSystem containing conditional model
        cond_model: The ConditionalModel (U-Net)
        config: Configuration dictionary
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the feature prior explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with settings
        """
        super().__init__(model, device, config)
        
        self.cond_model = model.model
        self.aux_model = model.aux_model
        self.cond_model.eval()
        self.aux_model.eval()
        
        self.num_classes = config.get('num_classes', 5)
        self.model_config = model.params
        
        print(f"[FeaturePriorExplainer] Initialized")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate feature prior explanation.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - 'explanation_type': 'feature_prior'
            - 'prediction': Predicted class
            - 'confidence': Confidence score
            - 'raw_features': Transformer encoder output
            - 'roi_features': CNN encoder outputs for patches
            - 'fusion_weights': Learnable fusion weights Q
            - 'fused_features': Final fused feature prior
            - 'contribution_scores': Contribution of each feature source
        """
        image = self._preprocess_image(image)
        
        # Get patches from aux model
        with torch.no_grad():
            (y_fusion, y_global, y_local,
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        # Extract features using conditional model encoders
        # Note: We need to hook into the model to get intermediate features
        raw_features, roi_features, fusion_weights = self._extract_features(
            image, patches, patch_attns
        )
        
        # Compute fused features
        fused_features = self._fuse_features(raw_features, roi_features, fusion_weights)
        
        # Compute contribution scores
        contribution_scores = self._compute_contributions(
            raw_features, roi_features, fusion_weights
        )
        
        # Get predictions
        probs_fusion = F.softmax(y_fusion, dim=1)[0]
        prediction = int(torch.argmax(probs_fusion, dim=0).item())
        confidence = float(probs_fusion[prediction].item())
        
        # Convert to numpy for storage
        # raw_features: [B, feature_dim] -> [feature_dim] for batch_size=1
        # roi_features: [B, N_p, feature_dim] -> [N_p, feature_dim] for batch_size=1
        # fused_features: [B, feature_dim] -> [feature_dim] for batch_size=1
        raw_features_np = raw_features[0].detach().cpu().numpy() if raw_features.shape[0] == 1 else raw_features.detach().cpu().numpy()
        roi_features_np = roi_features[0].detach().cpu().numpy() if roi_features.shape[0] == 1 else roi_features.detach().cpu().numpy()
        fused_features_np = fused_features[0].detach().cpu().numpy() if fused_features.shape[0] == 1 else fused_features.detach().cpu().numpy()
        
        explanation = self.get_explanation_dict(
            explanation_type='feature_prior',
            prediction=prediction,
            confidence=confidence,
            
            # Feature data
            raw_features=raw_features_np,
            roi_features=roi_features_np,
            fusion_weights=fusion_weights.detach().cpu().numpy(),
            fused_features=fused_features_np,
            contribution_scores=contribution_scores,
            
            # Feature statistics
            raw_feature_norm=float(np.linalg.norm(raw_features_np)),
            roi_feature_norm=float(np.linalg.norm(roi_features_np)),
            fusion_weight_norm=float(np.linalg.norm(fusion_weights.detach().cpu().numpy())),
            
            # Patch information
            num_patches=int(patches.shape[1]) if patches is not None else 0,
            patch_attention=patch_attns[0].detach().cpu().numpy() if patch_attns is not None and len(patch_attns) > 0 else None,
            
            # Ground truth
            ground_truth=label if label is not None else -1
        )
        
        return explanation
    
    def _extract_features(self,
                         image: torch.Tensor,
                         patches: torch.Tensor,
                         patch_attns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract raw and ROI features from encoders.
        
        Args:
            image: Input image [B, C, H, W]
            patches: Patch crops [B, N_p, I, J]
            patch_attns: Patch attention weights [B, N_p]
            
        Returns:
            Tuple of (raw_features, roi_features, fusion_weights)
        """
        # Get encoders from conditional model
        encoder_x = self.cond_model.encoder_x
        encoder_x_l = self.cond_model.encoder_x_l
        
        # Extract raw features (Transformer encoder)
        # Note: encoder_x returns [B, feature_dim] (flattened), not spatial features
        with torch.no_grad():
            raw_features = encoder_x(image)  # [B, feature_dim]
            
        # GroupNorm expects at least 3D: [B, C, ...]
        # Add spatial dimensions for normalization, then remove them
        if raw_features.ndim == 2:
            raw_features = raw_features.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
            raw_features = self.cond_model.norm(raw_features)
            raw_features = raw_features.squeeze(-1).squeeze(-1)  # [B, feature_dim]
        else:
            raw_features = self.cond_model.norm(raw_features)
        
        # Extract ROI features (CNN encoder)
        # Patches shape: [B, N_p, I, J] where I, J are patch height and width
        if patches is None:
            raise ValueError("Patches are None. Cannot extract ROI features.")
        
        bz, np, I, J = patches.shape
        
        # Move patches to device if needed
        patches = patches.to(self.device)
        
        # Reshape patches for encoder (same as in ConditionalModel.forward)
        # View to [B*N_p, I, J], add channel dim [B*N_p, 1, I, J], expand to 3 channels [B*N_p, 3, I, J]
        patches_reshaped = patches.contiguous().view(bz * np, I, J).unsqueeze(1)  # [B*N_p, 1, I, J]
        patches_reshaped = patches_reshaped.expand(-1, 3, -1, -1).contiguous()  # [B*N_p, 3, I, J]
        
        with torch.no_grad():
            roi_features = encoder_x_l(patches_reshaped)  # [B*N_p, feature_dim]
            
        # GroupNorm expects at least 3D: [B, C, ...]
        # The encoder returns 2D [B*N_p, feature_dim], so we need to add spatial dims for norm
        if roi_features.ndim == 2:
            # Add spatial dimensions: [B*N_p, feature_dim] -> [B*N_p, feature_dim, 1, 1]
            roi_features_4d = roi_features.unsqueeze(-1).unsqueeze(-1)
            roi_features_normalized = self.cond_model.norm_l(roi_features_4d)
            roi_features = roi_features_normalized.squeeze(-1).squeeze(-1)  # Back to [B*N_p, feature_dim]
        else:
            roi_features = self.cond_model.norm_l(roi_features)
        
        # Reshape ROI features back to batch format: [B*N_p, feature_dim] -> [B, N_p, feature_dim]
        roi_features = roi_features.view(bz, np, -1)
        
        # Get fusion weights
        fusion_weights = self.cond_model.cond_weight  # [1, feature_dim, 7]
        fusion_weights = F.softmax(fusion_weights, dim=2)  # Normalize
        
        return raw_features, roi_features, fusion_weights
    
    def _fuse_features(self,
                      raw_features: torch.Tensor,
                      roi_features: torch.Tensor,
                      fusion_weights: torch.Tensor) -> torch.Tensor:
        """
        Fuse raw and ROI features using learnable weights.
        
        This replicates the fusion process in ConditionalModel.forward().
        In the model: x (raw) is [B, feature_dim] and x_l (roi) is [B, N_p, feature_dim]
        They are concatenated: x = torch.cat([x.unsqueeze(-1), x_l.permute(0,2,1)], dim=-1)
        Then weighted sum: x_weight = torch.sum(x * w, dim=-1)
        
        Args:
            raw_features: Raw features [B, feature_dim]
            roi_features: ROI features [B, N_p, feature_dim]
            fusion_weights: Fusion weights [1, feature_dim, K+1] where K+1 = N_p+1
            
        Returns:
            Fused features [B, feature_dim]
        """
        bz = raw_features.shape[0]
        feature_dim = raw_features.shape[1]
        np = roi_features.shape[1]
        
        # Replicate the fusion from ConditionalModel.forward()
        # x_l is reshaped: [B, N_p, feature_dim] -> [B, feature_dim, N_p]
        roi_features_permuted = roi_features.permute(0, 2, 1)  # [B, feature_dim, N_p]
        
        # Concatenate raw and ROI features along last dimension
        # x.unsqueeze(-1) gives [B, feature_dim, 1]
        # x_l gives [B, feature_dim, N_p]
        # Concatenated: [B, feature_dim, N_p+1]
        raw_features_expanded = raw_features.unsqueeze(-1)  # [B, feature_dim, 1]
        concatenated_features = torch.cat([raw_features_expanded, roi_features_permuted], dim=-1)  # [B, feature_dim, N_p+1]
        
        # Apply fusion weights (already softmaxed)
        # w is [1, feature_dim, N_p+1], concatenated_features is [B, feature_dim, N_p+1]
        # Weighted sum: [B, feature_dim]
        fused = torch.sum(concatenated_features * fusion_weights, dim=-1)  # [B, feature_dim]
        
        return fused
    
    def _compute_contributions(self,
                              raw_features: torch.Tensor,
                              roi_features: torch.Tensor,
                              fusion_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute contribution scores for each feature source.
        
        Args:
            raw_features: Raw features
            roi_features: ROI features
            fusion_weights: Fusion weights
            
        Returns:
            Dictionary with contribution scores
        """
        # Compute feature magnitudes
        raw_magnitude = torch.norm(raw_features).item()
        roi_magnitude = torch.norm(roi_features).item()
        
        # Compute weight contributions
        # Fusion weights shape: [1, feature_dim, K+1]
        # First channel is for raw features, rest for ROIs
        weights = fusion_weights.squeeze(0)  # [feature_dim, K+1]
        raw_weight = weights[:, 0].mean().item()
        roi_weight = weights[:, 1:].mean().item()
        
        # Normalize contributions
        total_contribution = raw_magnitude * raw_weight + roi_magnitude * roi_weight
        if total_contribution > 0:
            raw_contribution = (raw_magnitude * raw_weight) / total_contribution
            roi_contribution = (roi_magnitude * roi_weight) / total_contribution
        else:
            raw_contribution = 0.5
            roi_contribution = 0.5
        
        return {
            'raw_contribution': float(raw_contribution),
            'roi_contribution': float(roi_contribution),
            'raw_magnitude': float(raw_magnitude),
            'roi_magnitude': float(roi_magnitude),
            'raw_weight': float(raw_weight),
            'roi_weight': float(roi_weight)
        }

