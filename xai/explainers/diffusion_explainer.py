"""
Diffusion Explainer Module

This module tracks and analyzes the diffusion denoising trajectory to understand
how predictions evolve through timesteps.

Key Insights:
- How do class probabilities change from noise to final prediction?
- Which timesteps are critical for decision-making?
- Is the model confident early or does it refine gradually?
- Do predictions oscillate or converge smoothly?

What We Track:
1. Intermediate predictions at each timestep (t=1000 -> 0)
2. Class probability trajectories
3. Prediction confidence over time
4. Convergence behavior
5. Critical timesteps where decisions solidify

Purpose:
- Understand the "reasoning process" of diffusion classification
- Identify interpretable stages in the denoising process
- Validate prediction stability
- Compare diffusion dynamics across samples

Future Extensions:
- Attention evolution through diffusion timesteps
- Guidance signal impact analysis
- Counterfactual trajectories (what if different initial noise?)
- Multi-sample trajectory clustering
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


class DiffusionExplainer(BaseExplainer):
    """
    Explain predictions by tracking the diffusion denoising trajectory.
    
    Unlike the auxiliary model which makes predictions in one shot,
    the diffusion model iteratively refines predictions over many timesteps.
    This explainer captures that iterative refinement process.
    
    At each timestep t, we have:
    - Noisy class probabilities y_t
    - Predicted noise
    - Denoised probabilities (after scheduler step)
    
    We track all of these to understand the prediction trajectory.
    
    Attributes:
        model: CoolSystem with diffusion sampler
        sampler: The SR3Sampler for denoising
        scheduler: The diffusion scheduler
        
    Usage:
        >>> explainer = DiffusionExplainer(model, device, config)
        >>> explanation = explainer.explain(image_tensor, label=2)
        >>> print(f"Tracked {len(explanation['trajectory'])} timesteps")
        >>> print(f"Final prediction: {explanation['prediction']}")
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the diffusion explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with diffusion settings
        """
        super().__init__(model, device, config)
        
        # Extract components
        self.aux_model = model.aux_model
        self.cond_model = model.model
        self.sampler = model.DiffSampler
        self.scheduler = self.sampler.scheduler
        self.model_config = model.params
        
        # Number of timesteps to track
        self.num_timesteps = len(self.scheduler.timesteps)
        
        # Whether to subsample timesteps for visualization
        self.subsample_timesteps = config.get('subsample_timesteps', None)
        
        print(f"[DiffusionExplainer] Initialized")
        print(f"  Total timesteps: {self.num_timesteps}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                track_all_timesteps: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Generate explanation by tracking diffusion trajectory.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            track_all_timesteps: If True, track every timestep. If False, subsample.
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
            - 'explanation_type': 'diffusion'
            - 'prediction': Final predicted class
            - 'confidence': Final confidence score
            - 'trajectory': List of predictions at each timestep
            - 'timesteps': Timestep values tracked
            - 'convergence_step': Timestep where prediction stabilizes
            - 'stability_score': Measure of prediction stability
            - 'initial_state': Starting noisy probabilities
            - 'final_state': Final denoised probabilities
        
        Process:
            1. Get auxiliary model outputs (guidance)
            2. Initialize noisy state
            3. Run denoising loop, tracking at each step
            4. Analyze trajectory
            5. Return comprehensive explanation
        """
        # Preprocess
        image = self._preprocess_image(image)
        
        # Get auxiliary model outputs for guidance
        with torch.no_grad():
            (y_fusion, y_global, y_local, 
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        # Prepare guidance conditions
        bz = image.shape[0]
        nc = self.model_config.data.num_classes
        
        # Get attention map size (assuming square)
        _, _, H, W = saliency_map.shape
        np_patches = int(np.sqrt(H * W))  # Number of patches (should be 7 for 7x7)
        
        # Create guided probability map
        y0_cond = self._guided_prob_map(y_global, y_local, bz, nc, np_patches)
        
        # Initialize with random noise (matching validation process)
        yT = self._guided_prob_map(
            torch.rand_like(y_global),
            torch.rand_like(y_local),
            bz, nc, np_patches
        )
        
        # Prepare attention
        attns = patch_attns.unsqueeze(-1)
        attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
        
        # Run diffusion with tracking
        trajectory = self._run_diffusion_with_tracking(
            image, yT, y0_cond, patches, attns,
            saliency_map, patch_attns,
            track_all=track_all_timesteps
        )
        
        # Analyze trajectory
        analysis = self._analyze_trajectory(trajectory)
        
        # Final prediction
        final_probs = trajectory[-1]['probs']
        prediction = int(np.argmax(final_probs))
        confidence = float(final_probs[prediction])
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='diffusion',
            prediction=prediction,
            confidence=confidence,
            
            # Trajectory data
            trajectory=trajectory,
            timesteps=[step['timestep'] for step in trajectory],
            
            # Analysis
            convergence_step=analysis['convergence_step'],
            stability_score=analysis['stability_score'],
            entropy_trajectory=analysis['entropy_trajectory'],
            confidence_trajectory=analysis['confidence_trajectory'],
            
            # Initial and final states
            initial_probs=trajectory[0]['probs'],
            final_probs=trajectory[-1]['probs'],
            
            # Auxiliary predictions for comparison
            aux_prediction=int(torch.argmax(y_fusion, dim=1).item()),
            aux_confidence=float(F.softmax(y_fusion, dim=1)[0].max().item()),
            
            # Attention evolution data (for animation)
            attention_evolution=[{
                'timestep': step['timestep'],
                'predicted_class': step.get('predicted_class', -1),
                'class_specific_saliency': step.get('class_specific_saliency', None),
                'patch_attention_weights': step.get('patch_attention_weights', None),
                'spatial_probs': step.get('spatial_probs', None),
            } for step in trajectory if 'class_specific_saliency' in step],
            
            # Ground truth
            ground_truth=label if label is not None else -1
        )
        
        return explanation
    
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
    
    def explain_with_hooks(self,
                           image: torch.Tensor,
                           label: Optional[int] = None,
                           track_timesteps: Optional[List[int]] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Generate explanation with forward hooks to track internal U-Net states.
        
        This method registers forward hooks on the ConditionalModel to extract
        intermediate features and attention weights at specified timesteps.
        
        Args:
            image: Input image tensor
            label: Ground truth label (optional)
            track_timesteps: List of timestep values to track (e.g., [900, 700, 500, 300, 100, 10])
                           If None, uses default timesteps
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing trajectory with internal states:
            - 'trajectory': List with 'cond_weights' and 'feature_maps' at tracked timesteps
        """
        # Default timesteps if not specified
        if track_timesteps is None:
            track_timesteps = [900, 700, 500, 300, 100, 10]
        
        # Preprocess
        image = self._preprocess_image(image)
        
        # Get auxiliary model outputs
        with torch.no_grad():
            (y_fusion, y_global, y_local, 
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        # Prepare guidance conditions
        bz = image.shape[0]
        nc = self.model_config.data.num_classes
        _, _, H, W = saliency_map.shape
        np_patches = int(np.sqrt(H * W))
        
        y0_cond = self._guided_prob_map(y_global, y_local, bz, nc, np_patches)
        yT = self._guided_prob_map(
            torch.rand_like(y_global),
            torch.rand_like(y_local),
            bz, nc, np_patches
        )
        
        attns = patch_attns.unsqueeze(-1)
        attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
        
        # Run diffusion with hooks
        trajectory = self._run_diffusion_with_hooks(
            image, yT, y0_cond, patches, attns,
            track_timesteps=track_timesteps
        )
        
        # Get standard explanation
        explanation = self.explain(image, label, track_all_timesteps=False)
        explanation['trajectory_with_hooks'] = trajectory
        
        return explanation
    
    def _run_diffusion_with_hooks(self,
                                  x_batch: torch.Tensor,
                                  yT: torch.Tensor,
                                  y0_cond: torch.Tensor,
                                  patches: torch.Tensor,
                                  attns: torch.Tensor,
                                  track_timesteps: List[int]) -> List[Dict]:
        """
        Run diffusion with forward hooks to capture internal states.
        
        Args:
            x_batch: Input image
            yT: Initial noisy state
            y0_cond: Guidance condition
            patches: Local patches
            attns: Attention weights
            track_timesteps: List of timestep values to track
            
        Returns:
            List of dictionaries with internal states at tracked timesteps
        """
        bz, nc, h, w = y0_cond.shape
        noisy_y = yT.clone()
        
        # Storage for hook outputs
        hook_storage = {}
        
        def forward_hook(module, input, output):
            """Hook on forward pass to extract cond_weight."""
            if hasattr(module, 'cond_weight'):
                w = torch.softmax(module.cond_weight, dim=2)
                hook_storage['cond_weight'] = w.detach().cpu().numpy()
        
        hook_handle = self.cond_model.register_forward_hook(forward_hook)
        
        trajectory = []
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            timesteps_batch = t * torch.ones(bz * h * w, dtype=t.dtype, device=x_batch.device)
            
            # Clear storage
            hook_storage.clear()
            
            with torch.no_grad():
                noise_pred = self.cond_model(
                    x_batch,
                    torch.cat([y0_cond, noisy_y], dim=1),
                    timesteps_batch,
                    patches,
                    attns
                )
            
            # Denoise
            prev_noisy_y = noisy_y.clone()
            noisy_y = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_y
            ).prev_sample
            
            # Check if this timestep should be tracked
            t_val = int(t.item())
            if t_val in track_timesteps:
                probs_spatial = F.softmax(noisy_y, dim=1)
                probs = probs_spatial.mean(dim=[2, 3])
                
                trajectory.append({
                    'timestep': t_val,
                    'timestep_idx': t_idx,
                    'probs': probs[0].detach().cpu().numpy(),
                    'predicted_class': int(torch.argmax(probs, dim=1).item()),
                    'cond_weights': hook_storage.get('cond_weight', None),
                    'spatial_probs': probs_spatial[0].detach().cpu().numpy(),
                })
        
        # Remove hook
        hook_handle.remove()
        
        return trajectory

    def _run_diffusion_with_tracking(self,
                                     x_batch: torch.Tensor,
                                     yT: torch.Tensor,
                                     y0_cond: torch.Tensor,
                                     patches: torch.Tensor,
                                     attns: torch.Tensor,
                                     saliency_map: torch.Tensor,
                                     patch_attns: torch.Tensor,
                                     track_all: bool = True) -> List[Dict]:
        """
        Run the diffusion denoising loop while tracking intermediate states.
        
        Args:
            x_batch: Input image
            yT: Initial noisy state
            y0_cond: Guidance condition
            patches: Local patches
            attns: Attention weights
            track_all: Whether to track all timesteps or subsample
        
        Returns:
            List of dictionaries, one per tracked timestep, containing:
            - timestep: The timestep value
            - noisy_y: Noisy probabilities before denoising
            - predicted_noise: Model's noise prediction
            - denoised_y: Probabilities after denoising step
            - probs: Class probabilities (averaged over spatial dimensions)
        """
        bz, nc, h, w = y0_cond.shape
        noisy_y = yT.clone()
        
        trajectory = []
        
        # Determine which timesteps to track
        if track_all:
            timesteps_to_track = self.scheduler.timesteps
        else:
            # Subsample for efficiency
            step = max(1, len(self.scheduler.timesteps) // 20)
            timesteps_to_track = self.scheduler.timesteps[::step]
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # Model prediction
            timesteps_batch = t * torch.ones(bz * h * w, dtype=t.dtype, device=x_batch.device)
            
            with torch.no_grad():
                noise_pred = self.cond_model(
                    x_batch,
                    torch.cat([y0_cond, noisy_y], dim=1),
                    timesteps_batch,
                    patches,
                    attns
                )
            
            # Denoise
            prev_noisy_y = noisy_y.clone()
            noisy_y = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_y
            ).prev_sample
            
            # Track this timestep?
            if t in timesteps_to_track or t_idx == 0 or t_idx == len(self.scheduler.timesteps) - 1:
                # Compute class probabilities by averaging spatial dimensions
                probs_spatial = F.softmax(noisy_y, dim=1)  # (1, nc, h, w)
                probs = probs_spatial.mean(dim=[2, 3])  # (1, nc)
                
                # Track attention evolution: compute class-specific attention influence
                # Weight saliency map by current class probabilities
                current_pred = torch.argmax(probs, dim=1).item()
                class_specific_saliency = saliency_map[0, current_pred].detach().cpu().numpy()
                
                # Compute attention-weighted probabilities (how much each patch contributes)
                patch_weights = patch_attns[0].detach().cpu().numpy()
                
                trajectory.append({
                    'timestep': int(t.item()),
                    'timestep_idx': t_idx,
                    'noisy_y': prev_noisy_y.detach().cpu().numpy(),
                    'predicted_noise': noise_pred.detach().cpu().numpy(),
                    'denoised_y': noisy_y.detach().cpu().numpy(),
                    'probs': probs[0].detach().cpu().numpy(),
                    'predicted_class': int(current_pred),
                    'class_specific_saliency': class_specific_saliency,
                    'patch_attention_weights': patch_weights,
                    'spatial_probs': probs_spatial[0].detach().cpu().numpy(),  # (nc, h, w)
                })
        
        return trajectory
    
    def _analyze_trajectory(self, trajectory: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the prediction trajectory.
        
        Args:
            trajectory: List of timestep dictionaries
        
        Returns:
            Analysis dictionary with:
            - convergence_step: Timestep where prediction stabilizes
            - stability_score: Overall stability (0=unstable, 1=stable)
            - entropy_trajectory: Entropy at each timestep
            - confidence_trajectory: Max probability at each timestep
        """
        # Extract probability trajectories
        probs_over_time = np.array([step['probs'] for step in trajectory])
        predictions_over_time = np.argmax(probs_over_time, axis=1)
        
        # Compute confidence (max probability) over time
        confidence_trajectory = np.max(probs_over_time, axis=1)
        
        # Compute entropy over time
        epsilon = 1e-10
        entropy_trajectory = -np.sum(
            probs_over_time * np.log(probs_over_time + epsilon),
            axis=1
        )
        
        # Find convergence step (where prediction stops changing)
        convergence_step = 0
        final_prediction = predictions_over_time[-1]
        for i in range(len(predictions_over_time) - 1, -1, -1):
            if predictions_over_time[i] != final_prediction:
                convergence_step = trajectory[i+1]['timestep'] if i+1 < len(trajectory) else trajectory[i]['timestep']
                break
        
        # Compute stability score (how consistent is the prediction?)
        prediction_changes = np.sum(predictions_over_time[1:] != predictions_over_time[:-1])
        stability_score = 1.0 - (prediction_changes / len(predictions_over_time))
        
        return {
            'convergence_step': int(convergence_step),
            'stability_score': float(stability_score),
            'entropy_trajectory': entropy_trajectory.tolist(),
            'confidence_trajectory': confidence_trajectory.tolist(),
            'num_prediction_changes': int(prediction_changes)
        }
    
    def compare_with_auxiliary(self,
                               diffusion_pred: int,
                               aux_pred: int,
                               ground_truth: Optional[int] = None) -> Dict[str, str]:
        """
        Compare diffusion vs auxiliary model predictions.
        
        Args:
            diffusion_pred: Prediction from diffusion model
            aux_pred: Prediction from auxiliary model
            ground_truth: True label (optional)
        
        Returns:
            Comparison summary
        
        Scenarios:
            1. Both correct: Strong agreement
            2. Both wrong, same: Consistent error
            3. Both wrong, different: Confused
            4. Diffusion correct, aux wrong: Diffusion refinement helps
            5. Aux correct, diffusion wrong: Diffusion refinement hurts
        """
        if ground_truth is None:
            return {
                'scenario': 'unknown',
                'description': 'Ground truth not provided',
                'agreement': diffusion_pred == aux_pred
            }
        
        diff_correct = (diffusion_pred == ground_truth)
        aux_correct = (aux_pred == ground_truth)
        agree = (diffusion_pred == aux_pred)
        
        if diff_correct and aux_correct:
            scenario = 'both_correct'
            description = 'Strong agreement - both models predict correctly'
        elif diff_correct and not aux_correct:
            scenario = 'diffusion_fixes_error'
            description = 'Diffusion refinement corrects auxiliary error'
        elif not diff_correct and aux_correct:
            scenario = 'diffusion_introduces_error'
            description = 'Diffusion refinement introduces error'
        elif not diff_correct and not aux_correct and agree:
            scenario = 'consistent_error'
            description = 'Both models make the same mistake'
        else:
            scenario = 'inconsistent_errors'
            description = 'Models make different mistakes'
        
        return {
            'scenario': scenario,
            'description': description,
            'agreement': agree,
            'diffusion_correct': diff_correct,
            'auxiliary_correct': aux_correct
        }

