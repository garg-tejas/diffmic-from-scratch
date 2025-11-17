"""
Conditional Attribution Explainer Module

This module computes attribution scores for the feature prior F and guidance map M
by backpropagating gradients through the entire reverse diffusion process.

Purpose:
- Answer: "Why did the model predict this DR grade?"
- Quantify reliance on global context vs. local lesions
- Prove which parts of the guidebook (DCG) the driver (U-Net) actually used

Method:
- Backpropagation Through Time (BPTT) through reverse diffusion
- Compute gradients of final prediction w.r.t. F (patches) and M (y0_cond)
- Aggregate to get global vs local contribution scores

This is the core module that transforms "useless" loggers into powerful attribution tools.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class ConditionalAttributionExplainer(BaseExplainer):
    """
    Compute attribution scores for conditional inputs (F and M).
    
    This explainer answers the critical question: "Which parts of the guidebook
    (DCG) did the driver (Denoising U-Net) actually read and listen to?"
    
    Attributes:
        model: CoolSystem model
        aux_model: Auxiliary DCG model
        cond_model: ConditionalModel (denoising U-Net)
        sampler: SR3Sampler for diffusion
        
    Usage:
        >>> explainer = ConditionalAttributionExplainer(model, device, config)
        >>> result = explainer.explain(image, label)
        >>> print(f"Global contribution: {result['global_contribution']:.2%}")
        >>> print(f"Local contribution: {result['local_contribution']:.2%}")
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize conditional attribution explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with attribution settings
        """
        super().__init__(model, device, config)
        
        # Extract components
        self.aux_model = model.aux_model
        self.cond_model = model.model
        self.sampler = model.DiffSampler
        self.scheduler = self.sampler.scheduler
        self.model_config = model.params
        
        # Enable gradients for attribution
        self._requires_grad = True
        
        print(f"[ConditionalAttributionExplainer] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Gradients enabled: {self._requires_grad}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Compute attribution scores via BPTT.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
            - 'prediction': Final predicted class
            - 'confidence': Final confidence
            - 'global_contribution': Attribution to F_raw (global prior)
            - 'local_contribution': Attribution to F_rois (local priors)
            - 'roi_contribution_scores': List of attribution per ROI
            - 'guidance_map_attribution': Attribution map for M (y0_cond)
            - 'dominant_roi': Index of most important ROI
            - 'guidance_strategy': 'global-dominant' or 'local-dominant'
        """
        # Preprocess
        image = self._preprocess_image(image)
        
        # Run attribution computation
        attribution_result = self.compute_attribution_via_bptt(image, label)
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='conditional_attribution',
            prediction=attribution_result['prediction'],
            confidence=attribution_result['confidence'],
            
            # Attribution scores
            global_contribution=attribution_result['global_contribution'],
            local_contribution=attribution_result['local_contribution'],
            roi_contribution_scores=attribution_result['roi_contribution_scores'],
            dominant_roi=attribution_result['dominant_roi'],
            
            # Guidance map attribution
            guidance_map_attribution=attribution_result['guidance_map_attribution'],
            guidance_strategy=attribution_result['guidance_strategy'],
            
            # Ground truth
            ground_truth=label if label is not None else -1,
        )
        
        return explanation
    
    def compute_attribution_via_bptt(self,
                                     image: torch.Tensor,
                                     label: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute attribution via Backpropagation Through Time.
        
        This is the core method that unrolls the reverse diffusion loop
        and computes gradients w.r.t. patches (F) and y0_cond (M).
        
        Args:
            image: Input image
            label: Ground truth label
        
        Returns:
            Attribution results dictionary
        """
        # Step 1: Get auxiliary model outputs
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(image)
        
        # Step 2: Create differentiable copies
        patches_grad = patches.detach().requires_grad_(True)
        y0_aux_global_grad = y0_aux_global.detach().requires_grad_(True)
        y0_aux_local_grad = y0_aux_local.detach().requires_grad_(True)
        
        # Prepare for diffusion
        bz = image.shape[0]
        nc = self.model_config.data.num_classes
        _, _, H, W = attn_map.shape
        np_patches = int(np.sqrt(H * W))
        
        # Create y0_cond from differentiable versions
        y0_cond_grad = self._guided_prob_map(
            y0_aux_global_grad, 
            y0_aux_local_grad, 
            bz, nc, np_patches
        )
        
        # Step 3: Unroll reverse diffusion manually with gradients
        yT = self._guided_prob_map(
            torch.rand_like(y0_aux_global),
            torch.rand_like(y0_aux_local),
            bz, nc, np_patches
        )
        
        # Prepare attention (keep original, not differentiable)
        attns_expanded = attns.unsqueeze(-1)
        attns_expanded = (attns_expanded * attns_expanded.transpose(1, 2)).unsqueeze(1)
        
        # Step 4: Manual reverse diffusion with gradient tracking
        noisy_y = yT.clone()
        intermediate_states = []  # Store states for backprop
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            timesteps_batch = t * torch.ones(int(bz * np_patches * np_patches), dtype=t.dtype, device=image.device)
            
            # Forward pass through U-Net (with gradients)
            y_fusion = torch.cat([y0_cond_grad, noisy_y], dim=1)
            
            noise_pred = self.cond_model(
                image,
                y_fusion,
                timesteps_batch,
                patches_grad,
                attns_expanded
            )
            
            # Denoise step
            prev_noisy_y = noisy_y.clone()
            noisy_y = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_y
            ).prev_sample
            
            # Store for backprop (keep in computation graph)
            intermediate_states.append({
                'noisy_y': noisy_y,
                'noise_pred': noise_pred,
            })
        
        # Step 5: Extract final prediction logit
        # Average over spatial dimensions
        final_probs_spatial = F.softmax(noisy_y, dim=1)  # (bz, nc, np, np)
        final_probs = final_probs_spatial.mean(dim=[2, 3])  # (bz, nc)
        prediction = int(torch.argmax(final_probs, dim=1).item())
        confidence = float(final_probs[0, prediction].item())
        
        # Get logit for predicted class
        final_logits_spatial = F.log_softmax(noisy_y, dim=1)  # Use logits for gradient
        final_logits = final_logits_spatial.mean(dim=[2, 3])  # (bz, nc)
        logit_pred = final_logits[0, prediction]
        
        # Step 6: Backpropagate gradients
        # Compute all gradients in one pass for efficiency
        grad_patches, grad_y0_cond, grad_global, grad_local = torch.autograd.grad(
            logit_pred,
            [patches_grad, y0_cond_grad, y0_aux_global_grad, y0_aux_local_grad],
            retain_graph=False,
            create_graph=False
        )
        
        # Step 7: Aggregate attribution scores
        # For patches (feature prior F): compute per-patch attribution
        # patches_grad shape: (bz, np, I, J)
        # grad_patches shape: (bz, np, I, J)
        patch_attribution_scores = torch.abs(grad_patches).sum(dim=[2, 3])  # (bz, np)
        patch_attribution_scores = patch_attribution_scores[0].detach().cpu().numpy()
        
        # For guidance map M: create attribution heatmap
        # grad_y0_cond shape: (bz, nc, np, np)
        guidance_map_attribution = torch.abs(grad_y0_cond[0]).mean(dim=0).detach().cpu().numpy()  # (np, np)
        
        # Determine guidance strategy and contributions from guidance map
        # In DiffMIC-v2, the guidance map M:
        # - Diagonal elements = global prediction (same class for all spatial locations)
        # - Off-diagonal elements = local prediction (varies by spatial location)
        np_size = guidance_map_attribution.shape[0]
        
        # Get diagonal and off-diagonal values
        diagonal_indices = np.arange(np_size)
        diagonal_values = guidance_map_attribution[diagonal_indices, diagonal_indices]
        
        # Create off-diagonal mask
        off_diagonal_mask = np.ones((np_size, np_size), dtype=bool)
        off_diagonal_mask[diagonal_indices, diagonal_indices] = False
        off_diagonal_values = guidance_map_attribution[off_diagonal_mask]
        
        # Use AVERAGE attribution (not sum) to avoid bias from different numbers of elements
        diagonal_attribution_avg = np.mean(diagonal_values)
        off_diagonal_attribution_avg = np.mean(off_diagonal_values)
        total_guidance_attribution = diagonal_attribution_avg + off_diagonal_attribution_avg
        
        # Global contribution = attribution to diagonal (global prediction)
        # Local contribution = attribution to off-diagonal (local predictions)
        if total_guidance_attribution > 0:
            global_contribution = diagonal_attribution_avg / total_guidance_attribution
            local_contribution = off_diagonal_attribution_avg / total_guidance_attribution
        else:
            global_contribution = 0.5
            local_contribution = 0.5
        
        # Determine strategy based on which dominates
        if global_contribution > local_contribution:
            guidance_strategy = 'global-dominant'
        else:
            guidance_strategy = 'local-dominant'
        
        # ROI contribution scores (from patch attribution)
        roi_contribution_scores = patch_attribution_scores.tolist()
        dominant_roi = int(np.argmax(patch_attribution_scores))
        
        # Cleanup
        torch.cuda.empty_cache()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'global_contribution': float(global_contribution),
            'local_contribution': float(local_contribution),
            'roi_contribution_scores': roi_contribution_scores,
            'dominant_roi': dominant_roi,
            'guidance_map_attribution': guidance_map_attribution,
            'guidance_strategy': guidance_strategy,
        }
    
    def _guided_prob_map(self, y0_g, y0_l, bz, nc, np_patches):
        """
        Create guided probability map (same as in training).
        
        This replicates the guided_prob_map method from CoolSystem.
        """
        device = y0_g.device
        
        # Create distance to diagonal matrix
        distance_to_diag = torch.tensor(
            [[abs(i-j) for j in range(np_patches)] for i in range(np_patches)]
        ).to(device)
        
        # Interpolation weights
        weight_g = 1 - distance_to_diag / (np_patches - 1)
        weight_l = distance_to_diag / (np_patches - 1)
        
        # Interpolate
        interpolated_value = (
            weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) +
            weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        )
        
        # Set diagonal and corners
        diag_indices = torch.arange(np_patches)
        prob_map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                prob_map[i, j, diag_indices, diag_indices] = y0_g[i, j]
                prob_map[i, j, np_patches-1, 0] = y0_l[i, j]
                prob_map[i, j, 0, np_patches-1] = y0_l[i, j]
        
        return prob_map

