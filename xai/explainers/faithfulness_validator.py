"""
Faithfulness Validator Module

This module validates the faithfulness of saliency maps and attention explanations
using insertion/deletion games and semantic robustness tests.

Purpose:
- Prove that saliency maps actually highlight pixels the model uses
- Quantify trustworthiness of explanations
- Test robustness to non-pathological changes

Methods:
1. Deletion Game: Occlude most important pixels, measure confidence drop
2. Insertion Game: Reveal only most important pixels, measure confidence rise
3. Semantic Robustness: Test stability under brightness/contrast/rotation changes

This replaces the simple NoiseExplainer with rigorous validation.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class FaithfulnessValidator(BaseExplainer):
    """
    Validate faithfulness of saliency maps using insertion/deletion games.
    
    This validator answers: "Is the saliency map actually correct?"
    It proves explanations are not just "plausible" but faithful to the model's logic.
    
    Attributes:
        model: CoolSystem model
        aux_model: Auxiliary DCG model (for saliency extraction)
        num_steps: Number of occlusion steps for deletion/insertion
        occlusion_method: 'blur' or 'mean' for pixel occlusion
        
    Usage:
        >>> validator = FaithfulnessValidator(model, device, config)
        >>> result = validator.explain(image, label, saliency_map)
        >>> print(f"Deletion AUC: {result['deletion_auc']:.3f}")
        >>> print(f"Insertion AUC: {result['insertion_auc']:.3f}")
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize faithfulness validator.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with validation settings
        """
        super().__init__(model, device, config)
        
        # Extract components
        self.aux_model = model.aux_model
        self.cond_model = model.model
        self.sampler = model.DiffSampler
        self.model_config = model.params
        
        # Configuration
        self.num_steps = config.get('deletion_steps', 20)
        self.occlusion_method = config.get('occlusion_method', 'blur')
        self.augmentation_count = config.get('augmentation_count', 10)
        self.blur_sigma = config.get('blur_sigma', 10.0)
        
        print(f"[FaithfulnessValidator] Initialized")
        print(f"  Occlusion method: {self.occlusion_method}")
        print(f"  Deletion steps: {self.num_steps}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                saliency_map: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Validate faithfulness of saliency map.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            saliency_map: Saliency map to validate (H, W) or (C, H, W)
                         If None, extracts from attention explainer
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
            - 'deletion_curve': Confidence vs occlusion percentage
            - 'insertion_curve': Confidence vs reveal percentage
            - 'deletion_auc': Area under deletion curve (lower is better)
            - 'insertion_auc': Area under insertion curve (higher is better)
            - 'robustness_scores': Dictionary of correlation scores
            - 'baseline_confidence': Original confidence
        """
        # Preprocess
        image = self._preprocess_image(image)
        
        # Get saliency map if not provided
        if saliency_map is None:
            saliency_map = self._extract_saliency_map(image, label)
        
        # Convert tensor to numpy if needed
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.detach().cpu().numpy()
        
        # Ensure saliency is numpy array
        saliency_map = np.asarray(saliency_map)
        
        # Ensure saliency is 2D
        if saliency_map.ndim == 3:
            # If (C, H, W), use class-specific or average
            if label is not None and label < saliency_map.shape[0]:
                saliency_map = saliency_map[label]
            else:
                saliency_map = saliency_map.mean(axis=0)
        elif saliency_map.ndim == 4:
            # If (1, C, H, W) or (B, C, H, W), squeeze batch and handle
            if saliency_map.shape[0] == 1:
                saliency_map = saliency_map[0]  # Remove batch dimension
            else:
                saliency_map = saliency_map[0]  # Take first batch
            
            # Now should be (C, H, W), handle as above
            if saliency_map.ndim == 3:
                if label is not None and label < saliency_map.shape[0]:
                    saliency_map = saliency_map[label]
                else:
                    saliency_map = saliency_map.mean(axis=0)
        
        # Get baseline prediction
        baseline_result = self._predict(image)
        baseline_confidence = baseline_result['confidence']
        predicted_class = baseline_result['prediction']
        
        # Run deletion game
        deletion_curve, deletion_auc = self.validate_deletion(
            image, saliency_map, predicted_class
        )
        
        # Run insertion game
        insertion_curve, insertion_auc = self.validate_insertion(
            image, saliency_map, predicted_class
        )
        
        # Run semantic robustness
        robustness_scores = self.validate_semantic_robustness(
            image, saliency_map, label
        )
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='faithfulness',
            prediction=predicted_class,
            confidence=baseline_confidence,
            
            # Deletion results
            deletion_curve=deletion_curve,
            deletion_auc=deletion_auc,
            
            # Insertion results
            insertion_curve=insertion_curve,
            insertion_auc=insertion_auc,
            
            # Robustness
            robustness_scores=robustness_scores,
            baseline_confidence=baseline_confidence,
            
            # Ground truth
            ground_truth=label if label is not None else -1,
            
            # Saliency map used
            saliency_map=saliency_map,
        )
        
        return explanation
    
    def validate_deletion(self,
                         image: torch.Tensor,
                         saliency_map: np.ndarray,
                         target_class: int) -> Tuple[List[Tuple[float, float]], float]:
        """
        Deletion game: Occlude most important pixels, measure confidence drop.
        
        Args:
            image: Input image
            saliency_map: Saliency map (H, W)
            target_class: Class to measure confidence for
        
        Returns:
            Tuple of (curve, auc) where:
            - curve: List of (occlusion_percentage, confidence) tuples
            - auc: Area under curve (lower is better - fast drop is good)
        """
        # Rank pixels by importance
        pixel_importance = saliency_map.flatten()
        pixel_indices = np.argsort(pixel_importance)[::-1]  # Most important first
        
        # Reshape for indexing
        h, w = saliency_map.shape
        pixel_coords = list(zip(*np.unravel_index(pixel_indices, (h, w))))
        
        # Baseline confidence
        baseline_result = self._predict(image)
        baseline_conf = float(baseline_result['probs'][target_class])
        
        curve = [(0.0, baseline_conf)]
        occluded_image = image.clone()
        
        # Occlude in steps
        total_pixels = h * w
        step_size = total_pixels // self.num_steps
        
        for step in range(1, self.num_steps + 1):
            # Calculate occlusion percentage
            num_occlude = step * step_size
            occlusion_pct = (num_occlude / total_pixels) * 100.0
            
            # Occlude only the NEW pixels in this step (incremental occlusion)
            prev_num_occlude = (step - 1) * step_size
            new_pixels_to_occlude = pixel_coords[prev_num_occlude:num_occlude]
            
            occluded_image = self._occlude_pixels(
                occluded_image,
                new_pixels_to_occlude,
                method=self.occlusion_method
            )
            
            # Measure confidence
            result = self._predict(occluded_image)
            confidence = float(result['probs'][target_class])
            
            curve.append((occlusion_pct, confidence))
        
        # Calculate AUC (normalized to [0, 1])
        # Lower AUC = faster drop = better faithfulness
        occlusion_pcts = [p[0] for p in curve]
        confidences = [p[1] for p in curve]
        auc = np.trapz(confidences, occlusion_pcts) / (100.0 * baseline_conf)
        
        return curve, float(auc)
    
    def validate_insertion(self,
                          image: torch.Tensor,
                          saliency_map: np.ndarray,
                          target_class: int) -> Tuple[List[Tuple[float, float]], float]:
        """
        Insertion game: Reveal only most important pixels, measure confidence rise.
        
        Args:
            image: Input image
            saliency_map: Saliency map (H, W)
            target_class: Class to measure confidence for
        
        Returns:
            Tuple of (curve, auc) where:
            - curve: List of (reveal_percentage, confidence) tuples
            - auc: Area under curve (higher is better - fast rise is good)
        """
        # Rank pixels by importance
        pixel_importance = saliency_map.flatten()
        pixel_indices = np.argsort(pixel_importance)[::-1]  # Most important first
        
        # Reshape for indexing
        h, w = saliency_map.shape
        pixel_coords = list(zip(*np.unravel_index(pixel_indices, (h, w))))
        
        # Start with fully blurred/black image
        if self.occlusion_method == 'blur':
            blurred_image = self._blur_image(image, sigma=50.0)
        else:
            blurred_image = torch.zeros_like(image)
            # Fill with mean color
            mean_color = image.mean(dim=[2, 3], keepdim=True)
            blurred_image = mean_color.expand_as(blurred_image)
        
        # Baseline confidence (should be low)
        baseline_result = self._predict(blurred_image)
        baseline_conf = float(baseline_result['probs'][target_class])
        
        curve = [(0.0, baseline_conf)]
        revealed_image = blurred_image.clone()
        
        # Reveal in steps
        total_pixels = h * w
        step_size = total_pixels // self.num_steps
        
        for step in range(1, self.num_steps + 1):
            # Calculate reveal percentage
            num_reveal = step * step_size
            reveal_pct = (num_reveal / total_pixels) * 100.0
            
            # Reveal only the NEW pixels in this step (incremental revelation)
            prev_num_reveal = (step - 1) * step_size
            new_pixels_to_reveal = pixel_coords[prev_num_reveal:num_reveal]
            
            revealed_image = self._reveal_pixels(
                revealed_image,
                image,
                new_pixels_to_reveal
            )
            
            # Measure confidence
            result = self._predict(revealed_image)
            confidence = float(result['probs'][target_class])
            
            curve.append((reveal_pct, confidence))
        
        # Calculate AUC (normalized to [0, 1])
        # Higher AUC = faster rise = better faithfulness
        reveal_pcts = [p[0] for p in curve]
        confidences = [p[1] for p in curve]
        
        # Get final confidence (should be close to original prediction)
        final_conf = confidences[-1]
        
        # Normalize by final confidence (or baseline if final is lower)
        normalizer = max(final_conf, baseline_conf)
        if normalizer > 0:
            auc = np.trapz(confidences, reveal_pcts) / (100.0 * normalizer)
        else:
            auc = 0.0
        
        return curve, float(auc)
    
    def validate_semantic_robustness(self,
                                    image: torch.Tensor,
                                    saliency_map: np.ndarray,
                                    label: Optional[int] = None) -> Dict[str, float]:
        """
        Test robustness to semantic perturbations (brightness, contrast, rotation).
        
        Args:
            image: Input image
            saliency_map: Original saliency map
            label: Ground truth label (optional)
        
        Returns:
            Dictionary with correlation scores for each augmentation
        """
        # Get original saliency (from attention explainer)
        original_saliency = self._extract_saliency_map(image, label)
        if original_saliency.ndim == 3:
            if label is not None and label < original_saliency.shape[0]:
                original_saliency = original_saliency[label]
            else:
                original_saliency = original_saliency.mean(axis=0)
        
        original_saliency_flat = original_saliency.flatten()
        
        robustness_scores = {}
        
        # Brightness perturbations
        brightness_corrs = []
        for delta in [-0.2, -0.1, 0.1, 0.2]:
            aug_image = self._adjust_brightness(image, delta)
            aug_saliency = self._extract_saliency_map(aug_image, label)
            if aug_saliency.ndim == 3:
                if label is not None and label < aug_saliency.shape[0]:
                    aug_saliency = aug_saliency[label]
                else:
                    aug_saliency = aug_saliency.mean(axis=0)
            
            corr, _ = pearsonr(original_saliency_flat, aug_saliency.flatten())
            brightness_corrs.append(corr)
        
        robustness_scores['brightness_correlation'] = float(np.mean(brightness_corrs))
        
        # Contrast perturbations
        contrast_corrs = []
        for factor in [0.8, 0.9, 1.1, 1.2]:
            aug_image = self._adjust_contrast(image, factor)
            aug_saliency = self._extract_saliency_map(aug_image, label)
            if aug_saliency.ndim == 3:
                if label is not None and label < aug_saliency.shape[0]:
                    aug_saliency = aug_saliency[label]
                else:
                    aug_saliency = aug_saliency.mean(axis=0)
            
            corr, _ = pearsonr(original_saliency_flat, aug_saliency.flatten())
            contrast_corrs.append(corr)
        
        robustness_scores['contrast_correlation'] = float(np.mean(contrast_corrs))
        
        # Rotation perturbations
        rotation_corrs = []
        for angle in [-5, -2, 2, 5]:
            aug_image = self._rotate_image(image, angle)
            aug_saliency = self._extract_saliency_map(aug_image, label)
            if aug_saliency.ndim == 3:
                if label is not None and label < aug_saliency.shape[0]:
                    aug_saliency = aug_saliency[label]
                else:
                    aug_saliency = aug_saliency.mean(axis=0)
            
            # Rotate saliency back for comparison
            aug_saliency_rotated = self._rotate_image(
                torch.from_numpy(aug_saliency).unsqueeze(0).unsqueeze(0),
                -angle
            ).squeeze().numpy()
            
            corr, _ = pearsonr(original_saliency_flat, aug_saliency_rotated.flatten())
            rotation_corrs.append(corr)
        
        robustness_scores['rotation_correlation'] = float(np.mean(rotation_corrs))
        robustness_scores['overall_robustness'] = float(np.mean([
            robustness_scores['brightness_correlation'],
            robustness_scores['contrast_correlation'],
            robustness_scores['rotation_correlation']
        ]))
        
        return robustness_scores
    
    def _extract_saliency_map(self, image: torch.Tensor, label: Optional[int] = None) -> np.ndarray:
        """Extract saliency map from auxiliary model."""
        with torch.no_grad():
            _, _, _, _, _, saliency_map = self.aux_model(image)
        
        # Convert to numpy
        saliency_map = saliency_map[0].detach().cpu().numpy()  # (nc, H, W)
        
        return saliency_map
    
    def _predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """Run full model prediction."""
        with torch.no_grad():
            # Get auxiliary outputs
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(image)
            
            # Prepare for diffusion
            bz, nc, H, W = attn_map.size()
            bz, np = attns.size()
            
            # Use the guided_prob_map method from CoolSystem
            y0_cond = self._guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np)
            yT = self._guided_prob_map(
                torch.rand_like(y0_aux_global),
                torch.rand_like(y0_aux_local),
                bz, nc, np
            )
            
            attns_expanded = attns.unsqueeze(-1)
            attns_expanded = (attns_expanded * attns_expanded.transpose(1, 2)).unsqueeze(1)
            
            # Run diffusion
            y_pred = self.sampler.sample_high_res(
                image,
                yT,
                conditions=[y0_cond, patches, attns_expanded]
            )
            
            # Average over spatial dimensions
            y_pred = y_pred.reshape(bz, nc, np * np)
            y_pred = y_pred.mean(2)  # (bz, nc)
            
            probs = F.softmax(y_pred, dim=1)
            prediction = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, prediction].item())
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probs': probs[0].detach().cpu().numpy(),
        }
    
    def _occlude_pixels(self, image: torch.Tensor, pixel_coords: List[Tuple], method: str = 'blur') -> torch.Tensor:
        """Occlude specified pixels in image."""
        # IMPORTANT: Work with the input image (which may already be occluded)
        # Don't reset to original - we want cumulative occlusion
        occluded = image.clone()
        
        if method == 'blur':
            # Use global mean color for blurring (more effective occlusion)
            # Computing this once from the current image state
            mean_color = image.mean(dim=[2, 3], keepdim=True)
            
            # Occlude pixels by setting to mean color (strong occlusion)
            for coord in pixel_coords:
                occluded[0, :, coord[0], coord[1]] = mean_color[0, :, 0, 0]
        else:  # 'mean'
            # Fill with mean color
            mean_color = image.mean(dim=[2, 3], keepdim=True)
            for coord in pixel_coords:
                occluded[0, :, coord[0], coord[1]] = mean_color[0, :, 0, 0]
        
        return occluded
    
    def _reveal_pixels(self, base_image: torch.Tensor, source_image: torch.Tensor, pixel_coords: List[Tuple]) -> torch.Tensor:
        """Reveal specified pixels from source image."""
        revealed = base_image.clone()
        
        for coord in pixel_coords:
            revealed[0, :, coord[0], coord[1]] = source_image[0, :, coord[0], coord[1]]
        
        return revealed
    
    def _blur_image(self, image: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
        """Apply Gaussian blur to image."""
        # Convert to numpy for scipy
        img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
        
        # Blur each channel
        blurred = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            blurred[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=sigma)
        
        # Convert back to tensor
        blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(image.device)
        
        return blurred_tensor
    
    def _adjust_brightness(self, image: torch.Tensor, delta: float) -> torch.Tensor:
        """Adjust image brightness."""
        return torch.clamp(image + delta, 0, 1)
    
    def _adjust_contrast(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust image contrast."""
        mean = image.mean()
        return torch.clamp((image - mean) * factor + mean, 0, 1)
    
    def _rotate_image(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate image by angle (degrees)."""
        import torchvision.transforms.functional as TF
        
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Rotate
        rotated = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        return rotated
    
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

