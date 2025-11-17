"""
Attention Explainer Module (AuxiliaryPriorValidator)

This module extracts and processes built-in attention mechanisms from the
auxiliary DCG (Deep Classification with Global-local features) model.

**Role in XAI-v2 Framework:**
This explainer serves as the "AuxiliaryPriorValidator" - it validates the DCG guidance
(the "guidebook"), not the final decision made by the Denoising U-Net. The output from
this module is the "Ground Truth Guidance Map" that we use to validate whether the U-Net's
final decision corresponds to the regions the DCG's saliency map highlighted.

**Important Note:**
This explainer answers "WHERE does the guidebook (DCG) get its information?" but does NOT
answer "Which parts of the guidebook did the driver (Denoising U-Net) actually read and
listen to when making its final decision?" For that, see ConditionalAttributionExplainer.

What We Extract:
1. Saliency Map (16x16): Global attention highlighting important regions
2. Patch Locations: Coordinates of 6 local regions of interest  
3. Patch Attention: Importance weights for each patch (6 values)
4. Global Prediction: Class probabilities from global pathway
5. Local Prediction: Class probabilities from local pathway

Purpose:
- Leverage model's built-in attention for explanations
- No additional computation needed (attention is part of forward pass)
- Provides multi-scale explanations (global + local)
- Validates DCG guidance quality (used in faithfulness validation)

Future Extensions:
- Extract attention from diffusion model's conditional weights
- Combine with gradient-based methods for richer explanations
- Temporal attention tracking across diffusion timesteps
- Class-specific attention decomposition
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class AttentionExplainer(BaseExplainer):
    """
    Extract built-in attention mechanisms from the auxiliary DCG model.
    
    The DCG model has two pathways:
    - Global: Processes full image, produces saliency map
    - Local: Processes 6 patches, produces patch-level predictions
    
    An attention module fuses these pathways using learned attention weights.
    
    This explainer extracts all attention artifacts to understand what
    the model focuses on when making predictions.
    
    Attributes:
        model: CoolSystem containing aux_model
        aux_model: The DCG auxiliary classifier
        config: Configuration dictionary
        
    Usage:
        >>> explainer = AttentionExplainer(model, device, config)
        >>> explanation = explainer.explain(image_tensor, label=2)
        >>> print(explanation.keys())
        dict_keys(['saliency_map', 'patch_locations', 'patch_attention', ...])
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the attention explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with settings for attention extraction
        """
        super().__init__(model, device, config)
        
        # Extract auxiliary model
        self.aux_model = model.aux_model
        self.aux_model.eval()
        
        # Config for preprocessing
        self.num_classes = config.get('num_classes', 5)
        self.model_config = model.params  # Original training config
        
        print(f"[AttentionExplainer] Initialized")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate attention-based explanation for a single image.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments (unused)
        
        Returns:
            Dictionary containing:
            - 'explanation_type': 'attention'
            - 'prediction': Predicted class
            - 'confidence': Confidence score
            - 'saliency_map': Global attention map (16, 16, num_classes)
            - 'patch_locations': Coordinates of patches (6, 2)
            - 'patch_attention': Attention weights for patches (6,)
            - 'global_prediction': Logits from global pathway
            - 'local_prediction': Logits from local pathway
            - 'fusion_prediction': Final fused prediction
            - 'patches': The actual patch images (6, H, W)
        
        Process:
            1. Preprocess image
            2. Forward pass through auxiliary model
            3. Extract all attention components
            4. Postprocess and return
        """
        # Preprocess
        image = self._preprocess_image(image)
        
        # Forward pass through auxiliary model
        with torch.no_grad():
            (y_fusion, y_global, y_local, 
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        # Get predictions
        fusion_probs = F.softmax(y_fusion, dim=1)
        prediction = torch.argmax(fusion_probs, dim=1).item()
        confidence = fusion_probs[0, prediction].item()
        
        # Build explanation dictionary
        explanation = self.get_explanation_dict(
            explanation_type='attention',
            prediction=prediction,
            confidence=confidence,
            
            # Attention artifacts
            saliency_map=saliency_map,  # (1, num_classes, 16, 16)
            patch_locations=self.aux_model.patch_locations,  # (1, 6, 2)
            patch_attention=patch_attns,  # (1, 6)
            
            # Predictions from different pathways
            global_prediction=y_global,  # (1, num_classes)
            local_prediction=y_local,  # (1, num_classes)
            fusion_prediction=y_fusion,  # (1, num_classes)
            
            # Patch images
            patches=patches,  # (1, 6, crop_h, crop_w)
            
            # Ground truth if provided
            ground_truth=label if label is not None else -1
        )
        
        # Postprocess (convert tensors to numpy)
        explanation = self._postprocess_explanation(explanation)
        
        # Add class-specific attention decomposition
        explanation['class_specific_attentions'] = self._decompose_class_attention(
            saliency_map, fusion_probs
        )
        
        return explanation
    
    def _decompose_class_attention(self, saliency_map: torch.Tensor, probs: torch.Tensor) -> Dict[int, np.ndarray]:
        """
        Decompose attention into class-specific components.
        
        For each class, compute how much attention is allocated to that class's features.
        This helps understand which regions are important for each class prediction.
        
        Args:
            saliency_map: Saliency map (1, num_classes, H, W) or (num_classes, H, W)
            probs: Class probabilities (1, num_classes)
        
        Returns:
            Dictionary mapping class_id -> attention map (H, W)
        """
        # Remove batch dimension if present
        if saliency_map.ndim == 4:
            saliency_map = saliency_map[0]  # (num_classes, H, W)
        
        num_classes = saliency_map.shape[0]
        class_attentions = {}
        
        # For each class, extract its saliency map
        for class_id in range(num_classes):
            # Get class-specific saliency
            class_saliency = saliency_map[class_id].detach().cpu().numpy()
            
            # Weight by class probability (how confident we are about this class)
            class_prob = probs[0, class_id].item()
            weighted_saliency = class_saliency * class_prob
            
            class_attentions[class_id] = weighted_saliency
        
        return class_attentions
    
    def get_class_specific_saliency(self, 
                                    saliency_map: np.ndarray,
                                    class_id: int) -> np.ndarray:
        """
        Extract saliency map for a specific class.
        
        Args:
            saliency_map: Full saliency map (num_classes, H, W)
            class_id: Class to extract
        
        Returns:
            Saliency map for specified class (H, W)
        
        Use Case:
            Understanding what the model looks at for different classes.
            E.g., "What regions suggest class 2 (Moderate DR)?"
            
        Future:
            - Contrast between predicted and ground truth class attention
            - Top-k discriminative regions across classes
            - Attention drift analysis (how attention changes with wrong predictions)
        """
        # Remove batch dim if present
        if saliency_map.ndim == 4:
            saliency_map = saliency_map[0]
        
        return saliency_map[class_id]
    
    def get_patch_importance_ranking(self, patch_attns: np.ndarray) -> np.ndarray:
        """
        Get patches ranked by importance.
        
        Args:
            patch_attns: Patch attention weights (6,) or (1, 6)
        
        Returns:
            Indices of patches sorted by importance (descending)
        
        Use Case:
            "Which patch contributed most to the prediction?"
            
        Example:
            >>> ranking = explainer.get_patch_importance_ranking(attns)
            >>> print(f"Most important patch: {ranking[0]}")
            >>> print(f"Least important patch: {ranking[-1]}")
        """
        # Remove batch dim if present
        if patch_attns.ndim == 2:
            patch_attns = patch_attns[0]
        
        # Sort in descending order
        return np.argsort(patch_attns)[::-1]
    
    def compute_attention_entropy(self, patch_attns: np.ndarray) -> float:
        """
        Compute entropy of patch attention distribution.
        
        Args:
            patch_attns: Patch attention weights (6,) or (1, 6)
        
        Returns:
            Entropy value (higher = more distributed attention)
        
        Interpretation:
            - Low entropy: Model focuses on few patches (confident, localized)
            - High entropy: Model considers many patches equally (uncertain, global)
            
        Future:
            - Compare entropy for correct vs incorrect predictions
            - Entropy as a confidence indicator
            - Temporal entropy changes through training
        """
        # Remove batch dim if present
        if patch_attns.ndim == 2:
            patch_attns = patch_attns[0]
        
        # Ensure it's a probability distribution
        if np.sum(patch_attns) == 0:
            return 0.0
        
        patch_attns = patch_attns / np.sum(patch_attns)
        
        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(patch_attns * np.log(patch_attns + epsilon))
        
        return float(entropy)
    
    def compare_global_local(self, 
                            global_pred: np.ndarray,
                            local_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compare predictions from global vs local pathways.
        
        Args:
            global_pred: Global pathway logits (num_classes,)
            local_pred: Local pathway logits (num_classes,)
        
        Returns:
            Dictionary with comparison metrics:
            - 'agreement': Whether they agree on prediction
            - 'global_class': Predicted class from global
            - 'local_class': Predicted class from local
            - 'kl_divergence': KL divergence between distributions
        
        Use Case:
            Understanding pathway disagreement:
            - If they agree: Strong evidence for prediction
            - If they disagree: Model is uncertain or finding different features
            
        Future:
            - Confidence gap analysis
            - Pathway reliability scoring
            - Error attribution (which pathway was wrong?)
        """
        # Remove batch dim if present
        if global_pred.ndim == 2:
            global_pred = global_pred[0]
        if local_pred.ndim == 2:
            local_pred = local_pred[0]
        
        # Get predicted classes
        global_class = int(np.argmax(global_pred))
        local_class = int(np.argmax(local_pred))
        agreement = (global_class == local_class)
        
        # Compute KL divergence
        global_probs = np.exp(global_pred) / np.sum(np.exp(global_pred))
        local_probs = np.exp(local_pred) / np.sum(np.exp(local_pred))
        
        # KL(global || local)
        epsilon = 1e-10
        kl_div = np.sum(global_probs * np.log((global_probs + epsilon) / (local_probs + epsilon)))
        
        return {
            'agreement': agreement,
            'global_class': global_class,
            'local_class': local_class,
            'kl_divergence': float(kl_div),
            'global_confidence': float(np.max(global_probs)),
            'local_confidence': float(np.max(local_probs))
        }
    
    def get_discriminative_regions(self,
                                  saliency_map: np.ndarray,
                                  top_k_percent: float = 0.2) -> np.ndarray:
        """
        Get the most discriminative regions (top-k% of saliency).
        
        Args:
            saliency_map: Saliency map (H, W) or (C, H, W)
            top_k_percent: Percentage of pixels to keep (0.0 to 1.0)
        
        Returns:
            Binary mask of discriminative regions (H, W)
        
        Use Case:
            Identifying the critical regions that drive the decision.
            Can be used for:
            - Guided attention visualizations
            - Occlusion-based validation
            - Region proposal for further analysis
            
        Future:
            - Connected component analysis to get discrete regions
            - Size filtering (ignore tiny regions)
            - Ranking regions by importance
        """
        # Handle multi-class saliency
        if saliency_map.ndim == 3:
            # Use max across classes
            saliency_map = np.max(saliency_map, axis=0)
        
        # Remove batch dim if present
        if saliency_map.ndim == 3:
            saliency_map = saliency_map[0]
        
        # Flatten and find threshold
        flat_saliency = saliency_map.flatten()
        threshold_idx = int(len(flat_saliency) * (1 - top_k_percent))
        threshold = np.sort(flat_saliency)[threshold_idx]
        
        # Create binary mask
        mask = saliency_map >= threshold
        
        return mask.astype(np.uint8)


"""
Usage Example:

from xai.core.model_loader import ModelLoader
from xai.explainers.attention_explainer import AttentionExplainer
from xai.utils.image_utils import load_image, preprocess_for_model

# Load model
loader = ModelLoader('logs/aptos/version_0/checkpoints/last.ckpt',
                     'configs/aptos.yml')
model = loader.load_model()

# Create explainer
config = {'num_classes': 5}
explainer = AttentionExplainer(model, loader.device, config)

# Load and preprocess image
image_pil = load_image('dataset/train_images/sample.png', target_size=(512, 512))
image_tensor = preprocess_for_model(image_pil)

# Generate explanation
explanation = explainer.explain(image_tensor, label=2)

# Access components
print(f"Prediction: {explanation['prediction']}")
print(f"Confidence: {explanation['confidence']:.3f}")
print(f"Saliency shape: {explanation['saliency_map'].shape}")
print(f"Patch attention: {explanation['patch_attention']}")

# Advanced analysis
ranking = explainer.get_patch_importance_ranking(explanation['patch_attention'])
entropy = explainer.compute_attention_entropy(explanation['patch_attention'])
comparison = explainer.compare_global_local(
    explanation['global_prediction'],
    explanation['local_prediction']
)

print(f"Most important patch: {ranking[0]}")
print(f"Attention entropy: {entropy:.3f}")
print(f"Global-local agreement: {comparison['agreement']}")

Future Research Directions:

1. Attention Calibration:
   - Are high-attention regions truly important?
   - Validate via occlusion experiments
   - Compare to human expert annotations

2. Attention Evolution:
   - Track attention changes during training
   - Identify when model "discovers" important features
   - Detect attention drift or instability

3. Multi-modal Attention:
   - Combine visual attention with other modalities
   - Cross-attention between global and local
   - Hierarchical attention at multiple scales

4. Attention-guided Interventions:
   - Use attention to guide data augmentation
   - Attention-based hard negative mining
   - Curriculum learning based on attention complexity
"""

