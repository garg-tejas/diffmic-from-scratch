"""
Noise Explainer Module

Analyzes heterologous noise patterns in 2D space during the diffusion process.

Features:
- Extracts noise samples at different timesteps
- Analyzes timestep distribution across spatial locations
- Tracks noise magnitude evolution
- Computes noise interaction patterns via convolution
- Provides spatial noise correlation maps
- Tracks noise reduction through denoising

Purpose:
- Understand how heterologous noise affects diffusion
- Analyze spatial noise patterns and interactions
- Visualize noise reduction process
- Support research on diffusion dynamics
- Enable noise pattern analysis and debugging

Future Extensions:
- Noise pattern clustering
- Noise-guided feature analysis
- Counterfactual noise experiments
- Multi-scale noise visualization
- Noise-sensitivity analysis
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class NoiseExplainer(BaseExplainer):
    """
    Analyze heterologous noise patterns in 2D space during diffusion.
    
    Heterologous noise means different noise levels at different spatial
    locations, creating a 2D noise map rather than uniform noise.
    
    Attributes:
        model: CoolSystem containing diffusion sampler
        cond_model: The ConditionalModel (U-Net)
        scheduler: The diffusion scheduler
        config: Configuration dictionary
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the noise explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with settings
        """
        super().__init__(model, device, config)
        
        self.cond_model = model.model
        self.sampler = model.DiffSampler
        self.scheduler = self.sampler.scheduler
        self.aux_model = model.aux_model
        self.aux_model.eval()
        
        self.num_classes = config.get('num_classes', 5)
        self.model_config = model.params
        
        self.num_steps_faithfulness = config.get('num_steps_faithfulness', 100)
        
        print(f"[NoiseExplainer] Initialized")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                track_all_timesteps: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Generate noise analysis explanation.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            track_all_timesteps: If True, track every timestep. If False, subsample.
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - 'explanation_type': 'noise'
            - 'prediction': Predicted class
            - 'confidence': Confidence score
            - 'noise_maps': List of noise maps at each timestep
            - 'timestep_maps': List of timestep distributions
            - 'noise_magnitudes': Noise magnitude evolution
            - 'noise_correlations': Spatial correlation patterns
        """
        image = self._preprocess_image(image)
        
        # Get auxiliary model outputs for guidance
        with torch.no_grad():
            (y_fusion, y_global, y_local,
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        bz = image.shape[0]
        nc = self.num_classes
        
        # Get patch count from saliency map dimensions
        _, _, H, W = saliency_map.shape
        np_patches = int(np.sqrt(H * W))
        
        # Create initial guidance condition
        y0_cond = self._guided_prob_map(y_global, y_local, bz, nc, np_patches)
        
        # Initialize with random noise (heterologous)
        yT = self._sample_heterologous_noise(bz, nc, np_patches)
        
        # Prepare attention
        attns = patch_attns.unsqueeze(-1)
        attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
        
        # Run diffusion with noise tracking
        noise_data = self._run_diffusion_with_noise_tracking(
            image, yT, y0_cond, patches, attns,
            track_all=track_all_timesteps
        )
        
        # Analyze noise patterns
        analysis = self._analyze_noise_patterns(noise_data)
        
        # Get final prediction
        final_probs = noise_data[-1]['probs']
        prediction = int(np.argmax(final_probs))
        confidence = float(final_probs[prediction])
        
        explanation = self.get_explanation_dict(
            explanation_type='noise',
            prediction=prediction,
            confidence=confidence,
            
            # Noise data
            noise_maps=[step['noise_map'] for step in noise_data],
            timestep_maps=[step.get('timestep_map', None) for step in noise_data],
            noise_magnitudes=[step['noise_magnitude'] for step in noise_data],
            noisy_features=[step['noisy_features'] for step in noise_data],
            
            # Analysis
            noise_correlations=analysis['correlations'],
            noise_reduction_curve=analysis['reduction_curve'],
            spatial_noise_pattern=analysis['spatial_pattern'],
            
            # Timesteps
            timesteps=[step['timestep'] for step in noise_data],
            
            # Ground truth
            ground_truth=label if label is not None else -1
        )
        
        return explanation
    
    def _sample_heterologous_noise(self, bz: int, nc: int, np_patches: int) -> torch.Tensor:
        """
        Sample heterologous noise with different levels at different spatial locations.
        
        Args:
            bz: Batch size
            nc: Number of classes
            np_patches: Number of patches (spatial dimension)
            
        Returns:
            Heterologous noise tensor [B, C, N_p, N_p]
        """
        device = self.device
        
        # Sample random noise for each spatial location
        # In practice, this could be correlated or have spatial structure
        noise = torch.randn(bz, nc, np_patches, np_patches, device=device)
        
        # Optionally add spatial correlation (convolve with small kernel)
        # This creates smooth noise patterns rather than pure random
        if np_patches > 1:
            # Create a small Gaussian kernel for smoothing
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
            kernel = kernel / (kernel_size * kernel_size)
            
            # Apply convolution to create spatially correlated noise
            noise_reshaped = noise.view(bz * nc, 1, np_patches, np_patches)
            noise_smooth = F.conv2d(noise_reshaped, kernel, padding=1)
            noise = noise_smooth.view(bz, nc, np_patches, np_patches)
        
        return noise
    
    def _guided_prob_map(self, y0_g: torch.Tensor, y0_l: torch.Tensor,
                        bz: int, nc: int, np_patches: int) -> torch.Tensor:
        """
        Create guided probability map (same as in DiffusionExplainer).
        
        Args:
            y0_g: Global prior logits [B, C]
            y0_l: Local prior logits [B, C]
            bz: Batch size
            nc: Number of classes
            np_patches: Number of patches
            
        Returns:
            Guidance map [B, C, N_p, N_p]
        """
        device = y0_g.device
        
        # Compute distance matrix
        distance_matrix = torch.tensor(
            [[abs(i - j) for j in range(np_patches)] for i in range(np_patches)],
            device=device,
            dtype=torch.float32
        )
        
        # Interpolation weights
        max_distance = np_patches - 1
        weight_g = 1 - distance_matrix / max_distance if max_distance > 0 else torch.ones_like(distance_matrix)
        weight_l = distance_matrix / max_distance if max_distance > 0 else torch.zeros_like(distance_matrix)
        
        # Convert to probabilities
        probs_g = F.softmax(y0_g, dim=1)
        probs_l = F.softmax(y0_l, dim=1)
        
        # Interpolate
        interpolated = (
            weight_l.unsqueeze(0).unsqueeze(0) * probs_l.unsqueeze(-1).unsqueeze(-1) +
            weight_g.unsqueeze(0).unsqueeze(0) * probs_g.unsqueeze(-1).unsqueeze(-1)
        )
        
        # Set diagonal and corners
        diag_indices = torch.arange(np_patches, device=device)
        prob_map = interpolated.clone()
        for i in range(bz):
            for j in range(nc):
                prob_map[i, j, diag_indices, diag_indices] = probs_g[i, j]
                prob_map[i, j, np_patches-1, 0] = probs_l[i, j]
                prob_map[i, j, 0, np_patches-1] = probs_l[i, j]
        
        return prob_map
    
    def _run_diffusion_with_noise_tracking(self,
                                          x_batch: torch.Tensor,
                                          yT: torch.Tensor,
                                          y0_cond: torch.Tensor,
                                          patches: torch.Tensor,
                                          attns: torch.Tensor,
                                          track_all: bool = True) -> List[Dict]:
        """
        Run diffusion while tracking noise patterns.
        
        Args:
            x_batch: Input image
            yT: Initial noisy state
            y0_cond: Guidance condition
            patches: Local patches
            attns: Attention weights
            track_all: Whether to track all timesteps
            
        Returns:
            List of dictionaries with noise data at each timestep
        """
        bz, nc, h, w = y0_cond.shape
        noisy_y = yT.clone()
        
        noise_data = []
        
        # Determine which timesteps to track
        if track_all:
            timesteps_to_track = self.scheduler.timesteps
        else:
            step = max(1, len(self.scheduler.timesteps) // 20)
            timesteps_to_track = self.scheduler.timesteps[::step]
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # Store previous state
            prev_noisy_y = noisy_y.clone()
            
            # Model prediction - match the actual diffusion process
            timesteps_batch = t * torch.ones(bz * h * w, dtype=t.dtype, device=x_batch.device)
            
            # Concatenate guidance and noisy state (matching training)
            y_fusion = torch.cat([y0_cond, prev_noisy_y], dim=1)
            
            with torch.no_grad():
                noise_pred = self.cond_model(
                    x_batch,
                    y_fusion,
                    timesteps_batch,
                    patches,
                    attns
                )
            
            # Scheduler step
            noisy_y = self.scheduler.step(noise_pred, t, prev_noisy_y).prev_sample
            
            # Track this timestep?
            if t in timesteps_to_track or t_idx == 0 or t_idx == len(self.scheduler.timesteps) - 1:
                # Compute noise map (difference between noisy and denoised)
                noise_map = (prev_noisy_y - noisy_y).abs()
                
                # Compute noise magnitude
                noise_magnitude = noise_map.mean(dim=1, keepdim=True)  # Average over classes
                
                # Compute probabilities
                probs_spatial = F.softmax(noisy_y, dim=1)
                probs = probs_spatial.mean(dim=[2, 3])
                
                # Sample timestep map (for heterologous noise visualization)
                # In practice, this would come from the actual sampling process
                timestep_map = torch.full((h, w), t.item(), device=x_batch.device)
                
                noise_data.append({
                    'timestep': int(t.item()),
                    'timestep_idx': t_idx,
                    'noise_map': noise_map[0].detach().cpu().numpy(),  # [C, H, W]
                    'noise_magnitude': noise_magnitude[0, 0].detach().cpu().numpy(),  # [H, W]
                    'noisy_features': noisy_y[0].detach().cpu().numpy(),  # [C, H, W]
                    'timestep_map': timestep_map.detach().cpu().numpy(),  # [H, W]
                    'probs': probs[0].detach().cpu().numpy(),  # [C]
                    'predicted_class': int(torch.argmax(probs, dim=1).item()),
                })
        
        return noise_data
    
    def _analyze_noise_patterns(self, noise_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze noise patterns across timesteps.
        
        Args:
            noise_data: List of noise data dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        if len(noise_data) == 0:
            return {
                'correlations': None,
                'reduction_curve': None,
                'spatial_pattern': None
            }
        
        # Compute noise reduction curve
        noise_magnitudes = [np.mean(step['noise_magnitude']) for step in noise_data]
        reduction_curve = np.array(noise_magnitudes)
        
        # Compute spatial correlation for first timestep
        if len(noise_data) > 0:
            first_noise = noise_data[0]['noise_magnitude']
            # Compute correlation matrix (simplified)
            # Flatten spatial dimensions and compute correlation
            if first_noise.ndim == 2:
                h, w = first_noise.shape
                flat_noise = first_noise.flatten()
                # Simple spatial correlation: correlate with neighbors
                correlations = np.corrcoef([
                    flat_noise,
                    np.roll(flat_noise, 1),  # Shift by 1
                    np.roll(flat_noise, w),  # Shift by width
                ])
            else:
                correlations = None
            
            spatial_pattern = first_noise
        else:
            correlations = None
            spatial_pattern = None
        
        return {
            'correlations': correlations,
            'reduction_curve': reduction_curve,
            'spatial_pattern': spatial_pattern
        }
    
    def inject_input_noise(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Inject input noise to image.
        
        Used for noise sensitivity analysis (Fig 5 & 6).
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            sigma: Noise level (standard deviation)
        
        Returns:
            Noisy image tensor with same shape as input
        """
        image = self._preprocess_image(image)
        
        # Generate noise
        noise = torch.randn_like(image) * sigma
        
        # Add noise to image
        noisy_image = image + noise
        
        # Clamp to valid range [0, 1] (assuming normalized image)
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return noisy_image
    
    def compute_noise_sensitivity(self,
                                  image: torch.Tensor,
                                  saliency_map: np.ndarray,
                                  sigma: float,
                                  n_runs: int = 20) -> Dict[str, Any]:
        """
        Compute noise sensitivity by running multiple times with input noise.
        
        Used for Fig 5: variance heatmap over N runs.
        
        Args:
            image: Input image tensor
            saliency_map: Baseline saliency map (H, W)
            sigma: Noise level
            n_runs: Number of runs for variance computation
        
        Returns:
            Dictionary containing:
            - 'clean_saliency': Baseline saliency map
            - 'noisy_saliencies': List of saliency maps under noise [n_runs]
            - 'mean_saliency': Mean saliency across runs
            - 'variance_map': Variance heatmap (H, W)
            - 'std_map': Standard deviation map (H, W)
        """
        image = self._preprocess_image(image)
        
        # Get baseline saliency (clean image)
        with torch.no_grad():
            y_fusion, _, _, _, _, saliency_map_clean = self.aux_model(image)
        
        # Remove batch dimension if present
        if isinstance(saliency_map_clean, torch.Tensor):
            saliency_map_clean = saliency_map_clean[0].detach().cpu().numpy()
        if saliency_map_clean.ndim == 3:
            # (num_classes, H, W) -> aggregate or use predicted class
            probs = F.softmax(y_fusion, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            saliency_map_clean = saliency_map_clean[pred]
        
        # Run multiple times with noise
        noisy_saliencies = []
        
        for run_idx in range(n_runs):
            # Inject noise
            noisy_image = self.inject_input_noise(image, sigma)
            
            # Get saliency on noisy image
            with torch.no_grad():
                y_fusion_noisy, _, _, _, _, saliency_map_noisy = self.aux_model(noisy_image)
            
            # Process saliency map
            if isinstance(saliency_map_noisy, torch.Tensor):
                saliency_map_noisy = saliency_map_noisy[0].detach().cpu().numpy()
            if saliency_map_noisy.ndim == 3:
                probs_noisy = F.softmax(y_fusion_noisy, dim=1)
                pred_noisy = torch.argmax(probs_noisy, dim=1).item()
                saliency_map_noisy = saliency_map_noisy[pred_noisy]
            
            noisy_saliencies.append(saliency_map_noisy)
        
        # Compute statistics
        noisy_saliencies_array = np.array(noisy_saliencies)  # [n_runs, H, W]
        mean_saliency = np.mean(noisy_saliencies_array, axis=0)
        variance_map = np.var(noisy_saliencies_array, axis=0)
        std_map = np.std(noisy_saliencies_array, axis=0)
        
        return {
            'clean_saliency': saliency_map_clean,
            'noisy_saliencies': noisy_saliencies,
            'mean_saliency': mean_saliency,
            'variance_map': variance_map,
            'std_map': std_map,
            'sigma': sigma,
            'n_runs': n_runs
        }
    
    def compute_prediction_inconsistency(self,
                                        model,
                                        dataset,
                                        noise_levels: list,
                                        batch_size: int = 2,
                                        n_runs_per_image: int = 5) -> Dict[str, Any]:
        """
        Compute prediction inconsistency (PI) across noise levels.
        
        Used for Fig 6: PI(σ) curve comparing with/without DCG.
        
        PI(σ) = (1/N) * Σ [predicted_label_i != predicted_label_clean]
        
        Args:
            model: CoolSystem model (or baseline without DCG)
            dataset: Dataset to evaluate
            noise_levels: List of noise levels [0.02, 0.04, 0.08]
            batch_size: Batch size for evaluation
            n_runs_per_image: Number of runs per image per noise level
        
        Returns:
            Dictionary containing:
            - 'noise_levels': List of noise levels tested
            - 'pi_scores': PI scores per noise level
            - 'per_image_results': Detailed results per image
        """
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        model.model.eval()
        model.aux_model.eval()
        
        # Store results
        all_clean_predictions = []
        all_noisy_predictions = {sigma: [] for sigma in noise_levels}
        all_labels = []
        
        print(f"Computing prediction inconsistency for {len(noise_levels)} noise levels...")
        
        # Process each batch
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="PI computation")):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get clean predictions
            with torch.no_grad():
                # Run diffusion for clean image
                y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = model.aux_model(images)
                
                bz, nc, H, W = attn_map.size()
                bz, np = attns.size()
                
                y0_cond = model.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np)
                yT = model.guided_prob_map(
                    torch.rand_like(y0_aux_global),
                    torch.rand_like(y0_aux_local),
                    bz, nc, np
                )
                
                attns = attns.unsqueeze(-1)
                attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
                
                y_pred_clean = model.DiffSampler.sample_high_res(
                    images, yT, conditions=[y0_cond, patches, attns]
                )
                y_pred_clean = y_pred_clean.reshape(bz, nc, np * np)
                y_pred_clean = y_pred_clean.mean(2)  # [batch_size, num_classes]
                probs_clean = F.softmax(y_pred_clean, dim=1)
                preds_clean = torch.argmax(probs_clean, dim=1).cpu().numpy()
            
            all_clean_predictions.extend(preds_clean)
            true_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
            all_labels.extend(true_labels.cpu().numpy())
            
            # Get noisy predictions for each noise level
            for sigma in noise_levels:
                noisy_preds_sigma = []
                
                for run_idx in range(n_runs_per_image):
                    # Inject noise
                    noisy_images = self.inject_input_noise(images, sigma)
                    
                    # Run diffusion on noisy image
                    with torch.no_grad():
                        y0_aux_noisy, y0_aux_global_noisy, y0_aux_local_noisy, patches_noisy, attns_noisy, attn_map_noisy = model.aux_model(noisy_images)
                        
                        bz_noisy, nc_noisy, H_noisy, W_noisy = attn_map_noisy.size()
                        bz_noisy, np_noisy = attns_noisy.size()
                        
                        y0_cond_noisy = model.guided_prob_map(y0_aux_global_noisy, y0_aux_local_noisy, bz_noisy, nc_noisy, np_noisy)
                        yT_noisy = model.guided_prob_map(
                            torch.rand_like(y0_aux_global_noisy),
                            torch.rand_like(y0_aux_local_noisy),
                            bz_noisy, nc_noisy, np_noisy
                        )
                        
                        attns_noisy = attns_noisy.unsqueeze(-1)
                        attns_noisy = (attns_noisy * attns_noisy.transpose(1, 2)).unsqueeze(1)
                        
                        y_pred_noisy = model.DiffSampler.sample_high_res(
                            noisy_images, yT_noisy, conditions=[y0_cond_noisy, patches_noisy, attns_noisy]
                        )
                        y_pred_noisy = y_pred_noisy.reshape(bz_noisy, nc_noisy, np_noisy * np_noisy)
                        y_pred_noisy = y_pred_noisy.mean(2)
                        probs_noisy = F.softmax(y_pred_noisy, dim=1)
                        preds_noisy = torch.argmax(probs_noisy, dim=1).cpu().numpy()
                    
                    noisy_preds_sigma.append(preds_noisy)
                
                # Store for this sigma
                all_noisy_predictions[sigma].extend(noisy_preds_sigma)
        
        # Compute PI scores
        all_clean_predictions = np.array(all_clean_predictions)
        pi_scores = []
        
        for sigma in noise_levels:
            # Flatten noisy predictions: [n_images * n_runs]
            noisy_preds_flat = np.array(all_noisy_predictions[sigma]).flatten()
            
            # Repeat clean predictions to match
            clean_preds_repeated = np.repeat(all_clean_predictions, n_runs_per_image)
            
            # Compute inconsistency: fraction of mismatches
            inconsistent = (noisy_preds_flat != clean_preds_repeated).mean()
            pi_scores.append(inconsistent)
        
        return {
            'noise_levels': noise_levels,
            'pi_scores': pi_scores,
            'n_runs_per_image': n_runs_per_image,
            'total_images': len(all_clean_predictions)
        }

