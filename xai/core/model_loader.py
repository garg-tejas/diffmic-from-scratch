"""
Model Loader Module

This module handles loading the trained diffusion-based classifier from checkpoints.
It provides a centralized interface for model loading with proper device handling.

Purpose:
- Load CoolSystem model from PyTorch Lightning checkpoints
- Reconstruct model architecture from configuration files
- Handle device placement (CPU/GPU) automatically
- Expose model components for XAI analysis

Key Components Exposed:
- main_model: The full CoolSystem (LightningModule)
- conditional_model: The diffusion denoising network
- aux_model: The auxiliary DCG classifier (provides attention)
- diffusion_sampler: The SR3Sampler for inference

Future Extensions:
- Support for loading from different checkpoint formats
- Model ensemble loading for uncertainty quantification
- Checkpoint validation and compatibility checking
- Automatic downloading of pretrained models
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import yaml
from easydict import EasyDict

# Add parent directory to path to import from main codebase
# Note: We don't import CoolSystem here to avoid triggering PyTorch Lightning imports
# at module level, which can cause dependency issues. Instead, we import it lazily
# inside the load_model method.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffuser_trainer import CoolSystem

class ModelLoader:
    """
    Handles loading trained diffusion classifier from checkpoints.
    
    This class provides a clean interface for loading the complete model
    with all its components. It handles the complexity of PyTorch Lightning
    checkpoints and ensures everything is properly initialized.
    
    Attributes:
        checkpoint_path: Path to the model checkpoint
        config: Model configuration (from aptos.yml)
        device: Device where model is loaded
        model: The loaded CoolSystem instance
        
    Usage:
        >>> loader = ModelLoader(checkpoint_path, config_path, device='cuda')
        >>> model = loader.load_model()
        >>> # Access components
        >>> aux_model = loader.get_aux_model()
        >>> diffusion_model = loader.get_conditional_model()
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: str,
                 device: Optional[str] = 'auto'):
        """
        Initialize the model loader.
        
        Args:
            checkpoint_path: Path to the .ckpt file (relative to project root)
            config_path: Path to the config .yml file used during training
            device: Device to load model on ('cuda', 'cpu', or 'auto')
                   'auto' will use CUDA if available, otherwise CPU
        
        Note:
            Paths are relative to the project root directory (parent of xai/)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.checkpoint_path = self.project_root / checkpoint_path
        self.config_path = self.project_root / config_path
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        # Initialize model container
        self.model: Optional[CoolSystem] = None
        self.config: Optional[EasyDict] = None
        
        print(f"[ModelLoader] Initialized")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Config: {self.config_path}")
        print(f"  Device: {self.device}")
    
    def load_model(self):
        """
        Load the complete model from checkpoint.
        
        This method:
        1. Loads the configuration file
        2. Loads the checkpoint
        3. Reconstructs the CoolSystem model
        4. Sets model to evaluation mode
        5. Moves model to specified device
        
        Returns:
            Loaded CoolSystem model in evaluation mode
        
        Raises:
            RuntimeError: If model loading fails
            ImportError: If dependencies (PyTorch Lightning) are not available
        
        Future Improvements:
            - Add checksum validation for checkpoints
            - Support for loading only specific components
            - Memory-efficient loading for large models
        """
        # Import CoolSystem (lazy import to avoid issues if not needed)
        try:
            from diffuser_trainer import CoolSystem
        except ImportError as e:
            raise ImportError(
                f"Failed to import CoolSystem from diffuser_trainer: {e}\n"
                f"Make sure you're running from the project root directory."
            )
        
        # Load configuration
        print(f"[ModelLoader] Loading config from {self.config_path}")
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = EasyDict(config_dict)
        
        # Load checkpoint
        print(f"[ModelLoader] Loading checkpoint from {self.checkpoint_path}")
        try:
            # PyTorch 2.6+ changed default weights_only to True, but our checkpoints contain EasyDict
            # Lightning's load_from_checkpoint uses torch.load internally, which now defaults to weights_only=True
            # We need to patch torch.load temporarily to use weights_only=False
            # This is safe since we trust our own checkpoints
            import functools
            original_load = torch.load
            
            @functools.wraps(original_load)
            def patched_load(*args, **kwargs):
                # Force weights_only=False for checkpoint loading
                # This allows loading checkpoints with EasyDict and other custom classes
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            # Temporarily replace torch.load
            torch.load = patched_load
            try:
                self.model = CoolSystem.load_from_checkpoint(
                    str(self.checkpoint_path),
                    hparams=self.config,
                    map_location=self.device
                )
            finally:
                # Restore original torch.load
                torch.load = original_load
            
            # Set to evaluation mode and move to device
            self.model.eval()
            self.model.to(self.device)
            
            # Disable gradients for all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f"[ModelLoader] [OK] Model loaded successfully")
            print(f"[ModelLoader]   - Device: {self.device}")
            print(f"[ModelLoader]   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return self.model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_model(self):
        """
        Get the loaded model instance.
        
        Returns:
            The loaded CoolSystem model
        
        Raises:
            RuntimeError: If model hasn't been loaded yet
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_aux_model(self) -> torch.nn.Module:
        """
        Get the auxiliary DCG model.
        
        The auxiliary model provides:
        - Saliency maps (16x16 attention)
        - Patch locations (6 regions of interest)
        - Patch attention weights
        - Global and local predictions
        
        Returns:
            The auxiliary DCG model
        
        Usage:
            >>> aux_model = loader.get_aux_model()
            >>> with torch.no_grad():
            ...     y_fusion, y_global, y_local, patches, attns, saliency = aux_model(x)
        
        Future:
            - Add methods to extract specific components
            - Support for different auxiliary model architectures
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.aux_model
    
    def get_conditional_model(self) -> torch.nn.Module:
        """
        Get the conditional diffusion model.
        
        This is the main denoising network that predicts noise at each timestep.
        It takes:
        - Input image
        - Noisy class probabilities
        - Timestep
        - Patch features
        - Attention weights
        
        Returns:
            The ConditionalModel (denoising U-Net)
        
        Usage:
            >>> cond_model = loader.get_conditional_model()
            >>> noise_pred = cond_model(x_batch, y_fusion, timesteps, patches, attns)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.model
    
    def get_diffusion_sampler(self):
        """
        Get the diffusion sampler for inference.
        
        The sampler handles the denoising process:
        - Takes noisy initial state
        - Iteratively denoises through timesteps
        - Returns final class probability predictions
        
        Returns:
            The SR3Sampler instance
        
        Usage:
            >>> sampler = loader.get_diffusion_sampler()
            >>> y_pred = sampler.sample_high_res(x, yT, conditions=[y0_cond, patches, attns])
        
        Note:
            For XAI, we often want to modify the sampling loop to extract
            intermediate states. This can be done by accessing sampler.scheduler
            and manually iterating through timesteps.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.DiffSampler
    
    def get_config(self) -> EasyDict:
        """
        Get the model configuration.
        
        Returns:
            Configuration dictionary
        """
        if self.config is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.config
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing:
            - num_parameters: Total number of parameters
            - num_classes: Number of output classes
            - image_size: Expected input image size
            - diffusion_timesteps: Number of diffusion steps
            - device: Current device
        
        Future:
            - Add model architecture summary
            - Include training metrics from checkpoint
            - Memory usage information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_classes': self.config.data.num_classes,
            'image_size': self.config.data.get('image_size', 512),
            'diffusion_timesteps': self.config.diffusion.timesteps,
            'test_timesteps': self.config.diffusion.test_timesteps,
            'device': str(self.device),
            'checkpoint_path': str(self.checkpoint_path),
            'architecture': self.config.model.arch,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.model is not None else "not loaded"
        return f"ModelLoader(checkpoint={self.checkpoint_path.name}, status={status}, device={self.device})"


"""
Usage Example:

# In main XAI pipeline:
from core.model_loader import ModelLoader

# Initialize loader
loader = ModelLoader(
    checkpoint_path='logs/aptos/version_0/checkpoints/last.ckpt',
    config_path='configs/aptos.yml',
    device='auto'
)

# Load model
model = loader.load_model()

# Get components for XAI
aux_model = loader.get_aux_model()
cond_model = loader.get_conditional_model()
sampler = loader.get_diffusion_sampler()

# Use in explainers
attention_explainer = AttentionExplainer(model, loader.device, config)
diffusion_explainer = DiffusionExplainer(model, loader.device, config)

Future Extensions:
- Model versioning and compatibility checking
- Support for distributed model loading (multi-GPU)
- Lazy loading of model components
- Model quantization for faster inference
- Checkpoint averaging for better performance
"""

