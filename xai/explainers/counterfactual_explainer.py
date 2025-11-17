"""
Generative Counterfactual Explainer Module

This module generates counterfactual examples by guiding the diffusion process
toward a target class using the auxiliary DCG model as a classifier.

Purpose:
- Answer: "What minimal visual evidence would change the prediction?"
- Generate counterfactual images showing what makes Grade X → Grade Y
- Visualize the delta (difference) between original and counterfactual

Method:
- Use guided diffusion with DCG model as classifier
- At each timestep, compute gradient toward target class
- Steer the denoising process to generate counterfactual
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


class GenerativeCounterfactualExplainer(BaseExplainer):
    """
    Generate counterfactual examples using guided diffusion.
    
    This explainer uses the auxiliary DCG model to guide the reverse diffusion
    process toward a target class, generating a counterfactual image that shows
    what features would need to change to flip the prediction.
    
    Attributes:
        model: CoolSystem model
        aux_model: Auxiliary DCG model (used as classifier for guidance)
        cond_model: ConditionalModel (denoising U-Net)
        sampler: SR3Sampler
        
    Usage:
        >>> explainer = GenerativeCounterfactualExplainer(model, device, config)
        >>> result = explainer.generate_counterfactual(image, current_class=1, target_class=2)
        >>> print(f"Counterfactual generated: {result['counterfactual_image'].shape}")
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize counterfactual explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with counterfactual settings
        """
        super().__init__(model, device, config)
        
        # Extract components
        self.aux_model = model.aux_model
        self.cond_model = model.model
        self.sampler = model.DiffSampler
        self.scheduler = self.sampler.scheduler
        self.model_config = model.params
        
        # Configuration
        self.default_guidance_scale = config.get('guidance_scale', 5.0)
        
        print(f"[GenerativeCounterfactualExplainer] Initialized")
        print(f"  Default guidance scale: {self.default_guidance_scale}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                target_class: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate counterfactual explanation.
        
        Args:
            image: Input image tensor
            label: Current class (ground truth or prediction)
            target_class: Target class for counterfactual (if None, uses opposite)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
            - 'counterfactual_image': Generated counterfactual
            - 'delta_map': Pixel-wise difference map
            - 'original_prediction': Original prediction
            - 'counterfactual_prediction': Counterfactual prediction
            - 'trajectory': Denoising trajectory
        """
        # Preprocess
        image = self._preprocess_image(image)
        
        # Get current prediction
        with torch.no_grad():
            current_result = self._predict(image)
            current_class = current_result['prediction']
        
        # Determine target class
        if target_class is None:
            # Default: try to flip to adjacent class
            num_classes = self.model_config.data.num_classes
            if current_class < num_classes - 1:
                target_class = current_class + 1
            else:
                target_class = current_class - 1
        
        # Generate counterfactual
        result = self.generate_counterfactual(
            image, current_class, target_class, self.default_guidance_scale
        )
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='counterfactual',
            prediction=current_class,
            confidence=current_result['confidence'],
            
            # Counterfactual results
            counterfactual_image=result['counterfactual_image'],
            delta_map=result['delta_map'],
            original_prediction=current_class,
            counterfactual_prediction=result['counterfactual_prediction'],
            target_class=target_class,
            trajectory=result.get('trajectory', []),
            
            # Ground truth
            ground_truth=label if label is not None else -1,
        )
        
        return explanation
    
    def generate_counterfactual(self,
                                image: torch.Tensor,
                                current_class: int,
                                target_class: int,
                                guidance_scale: float = 5.0) -> Dict[str, Any]:
        """
        Generate counterfactual by guiding diffusion toward target class.
        
        Args:
            image: Original image
            current_class: Current predicted class
            target_class: Target class for counterfactual
            guidance_scale: Strength of guidance (higher = stronger steering)
        
        Returns:
            Counterfactual generation results
        """
        # Get auxiliary model outputs for original image
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(image)
        
        # Prepare for diffusion (use patch grid size derived from attention weights for consistency)
        bz = image.shape[0]
        nc = self.model_config.data.num_classes
        _, np_patches = attns.size()  # grid dimension inferred from attention weights
        
        y0_cond = self._guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np_patches)
        
        # Initialize with random noise
        yT = self._guided_prob_map(
            torch.rand_like(y0_aux_global),
            torch.rand_like(y0_aux_local),
            bz, nc, np_patches
        )
        
        attns_expanded = attns.unsqueeze(-1)
        attns_expanded = (attns_expanded * attns_expanded.transpose(1, 2)).unsqueeze(1)
        
        # Guided reverse diffusion
        noisy_y = yT.clone()
        trajectory = []
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            timesteps_batch = t * torch.ones(int(bz * np_patches * np_patches), dtype=t.dtype, device=image.device)
            
            # Standard denoising step
            with torch.no_grad():
                noise_pred = self.cond_model(
                    image,
                    torch.cat([y0_cond, noisy_y], dim=1),
                    timesteps_batch,
                    patches,
                    attns_expanded
                )
            
            # Denoise
            prev_noisy_y = noisy_y.clone()
            noisy_y = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_y
            ).prev_sample
            
            # Guidance step: steer toward target class
            if t_idx % 5 == 0:  # Apply guidance every 5 steps for efficiency
                # Enable gradients for guidance
                noisy_y_grad = noisy_y.requires_grad_(True)
                probs_grad = F.softmax(noisy_y_grad.mean(dim=[2, 3]), dim=1)
                target_logit_grad = probs_grad[0, target_class]
                
                # Compute gradient toward target class
                grad = torch.autograd.grad(
                    target_logit_grad,
                    noisy_y_grad,
                    retain_graph=False,
                    create_graph=False
                )[0]
                
                # Steer toward target
                noisy_y = noisy_y + guidance_scale * grad * 0.01
                
                # Clamp to valid range
                noisy_y = torch.clamp(noisy_y, -1.0, 1.0)
            
            # Track trajectory (subsample)
            if t_idx % 20 == 0 or t_idx == len(self.scheduler.timesteps) - 1:
                probs_spatial = F.softmax(noisy_y, dim=1)
                probs = probs_spatial.mean(dim=[2, 3])
                prediction = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, prediction].item())
                
                trajectory.append({
                    'timestep': int(t.item()),
                    'timestep_idx': t_idx,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probs': probs[0].detach().cpu().numpy(),
                })
        
        # Final counterfactual prediction
        final_probs_spatial = F.softmax(noisy_y, dim=1)
        final_probs = final_probs_spatial.mean(dim=[2, 3])
        counterfactual_prediction = int(torch.argmax(final_probs, dim=1).item())
        
        # Get original prediction for comparison (run standard inference)
        with torch.no_grad():
            orig_y0_aux, orig_y0_aux_global, orig_y0_aux_local, orig_patches, orig_attns, orig_attn_map = self.aux_model(image)
            orig_bz, orig_nc, orig_H, orig_W = orig_attn_map.shape
            orig_np = orig_attns.size()[1]
            orig_y0_cond = self._guided_prob_map(orig_y0_aux_global, orig_y0_aux_local, orig_bz, orig_nc, orig_np)
            orig_yT = self._guided_prob_map(
                torch.rand_like(orig_y0_aux_global),
                torch.rand_like(orig_y0_aux_local),
                orig_bz, orig_nc, orig_np
            )
            orig_attns_expanded = orig_attns.unsqueeze(-1)
            orig_attns_expanded = (orig_attns_expanded * orig_attns_expanded.transpose(1, 2)).unsqueeze(1)
            orig_y_pred = self.sampler.sample_high_res(
                image,
                orig_yT,
                conditions=[orig_y0_cond, orig_patches, orig_attns_expanded]
            )
            orig_probs_spatial = F.softmax(orig_y_pred, dim=1)  # (bz, nc, np, np)
        
        # Counterfactual is the probability map showing what the model predicts after guidance
        counterfactual_probs = final_probs_spatial[0].detach().cpu().numpy()  # (nc, np, np)
        original_probs = orig_probs_spatial[0].detach().cpu().numpy()  # (nc, np, np)
        
        # Delta map: difference in probability maps (shows what changed in model's reasoning)
        delta_map = np.abs(counterfactual_probs - original_probs)
        delta_map = delta_map.max(axis=0)  # Max over classes to get overall change
        
        return {
            'counterfactual_image': counterfactual_probs,
            'delta_map': delta_map,
            'counterfactual_prediction': counterfactual_prediction,
            'trajectory': trajectory,
        }
    
    def _predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """Run full model prediction."""
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(image)
            
            bz, nc, H, W = attn_map.size()
            bz, np = attns.size()
            
            y0_cond = self._guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np)
            yT = self._guided_prob_map(
                torch.rand_like(y0_aux_global),
                torch.rand_like(y0_aux_local),
                bz, nc, np
            )
            
            attns_expanded = attns.unsqueeze(-1)
            attns_expanded = (attns_expanded * attns_expanded.transpose(1, 2)).unsqueeze(1)
            
            y_pred = self.sampler.sample_high_res(
                image,
                yT,
                conditions=[y0_cond, patches, attns_expanded]
            )
            
            y_pred = y_pred.reshape(bz, nc, np * np)
            y_pred = y_pred.mean(2)
            
            probs = F.softmax(y_pred, dim=1)
            prediction = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, prediction].item())
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probs': probs[0].detach().cpu().numpy(),
        }
    
    def _guided_prob_map(self, y0_g, y0_l, bz, nc, np_patches):
        """Create guided probability map."""
        device = y0_g.device
        
        distance_to_diag = torch.tensor(
            [[abs(i-j) for j in range(np_patches)] for i in range(np_patches)]
        ).to(device)
        
        weight_g = 1 - distance_to_diag / (np_patches - 1)
        weight_l = distance_to_diag / (np_patches - 1)
        
        interpolated_value = (
            weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) +
            weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        )
        
        diag_indices = torch.arange(np_patches)
        prob_map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                prob_map[i, j, diag_indices, diag_indices] = y0_g[i, j]
                prob_map[i, j, np_patches-1, 0] = y0_l[i, j]
                prob_map[i, j, 0, np_patches-1] = y0_l[i, j]
        
        return prob_map

