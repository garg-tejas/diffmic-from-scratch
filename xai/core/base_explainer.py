"""
Base Explainer Module

This module defines the abstract base class that all XAI methods must inherit from.
This ensures a consistent interface across different explanation techniques.

Purpose:
- Standardize the API for all explainers
- Enable easy addition of new XAI methods
- Facilitate comparison between different explanation techniques

Future Extensions:
- Add methods for explanation validation (faithfulness, stability)
- Support for batch processing optimizations
- Caching mechanisms for expensive computations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import numpy as np


class BaseExplainer(ABC):
    """
    Abstract base class for all XAI methods.
    
    All explainers must implement the core methods defined here. This ensures
    that different XAI techniques can be used interchangeably in the pipeline.
    
    Attributes:
        model: The trained model to explain
        device: Device to run computations on (cuda/cpu)
        config: Configuration dictionary with explainer-specific settings
    
    Design Philosophy:
        - Each explainer should be self-contained and stateless
        - Explanations should be returned in a standardized dictionary format
        - All explainers should handle both single images and batches
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the base explainer.
        
        Args:
            model: The trained PyTorch model to generate explanations for
            device: Device to run computations on
            config: Configuration dictionary with settings specific to this explainer
        
        Note:
            Subclasses should call super().__init__() and then initialize
            any explainer-specific components
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation by default (can be enabled in subclasses if needed)
        self._requires_grad = False
    
    @abstractmethod
    def explain(self, 
                image: torch.Tensor, 
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate explanation for a single image.
        
        This is the core method that each explainer must implement. It takes
        an image and optionally a label, and returns a dictionary containing
        all explanation artifacts.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional, used for label-specific explanations)
            **kwargs: Additional explainer-specific arguments
        
        Returns:
            Dictionary containing explanation artifacts. Must include:
            - 'explanation_type': str, name of the explanation method
            - 'prediction': int, predicted class
            - 'confidence': float, confidence score for prediction
            
            Additional keys depend on the specific explainer (e.g., 'attention_map',
            'saliency', 'trajectory', etc.)
        
        Example:
            >>> explainer = SomeExplainer(model, device, config)
            >>> image = torch.randn(3, 512, 512)
            >>> explanation = explainer.explain(image)
            >>> print(explanation.keys())
            dict_keys(['explanation_type', 'prediction', 'confidence', 'saliency_map'])
        
        Future Work:
            - Add support for multi-label classification
            - Include uncertainty estimates in output
            - Support for returning intermediate layer activations
        """
        pass
    
    def batch_explain(self, 
                     images: torch.Tensor, 
                     labels: Optional[List[int]] = None,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of images.
        
        Default implementation processes images one at a time. Subclasses can
        override this for more efficient batch processing if the explanation
        method supports it.
        
        Args:
            images: Batch of images of shape (B, C, H, W)
            labels: List of ground truth labels (optional)
            **kwargs: Additional explainer-specific arguments
        
        Returns:
            List of explanation dictionaries, one per image
        
        Note:
            Some XAI methods (like LIME, SHAP) are inherently sequential and
            won't benefit from batch processing. Others (like gradient-based
            methods) can be significantly accelerated with batching.
        
        Future Optimization:
            - Implement intelligent batching based on explainer type
            - Add progress tracking for long batch operations
            - Support for distributed processing across multiple GPUs
        """
        if labels is None:
            labels = [None] * len(images)
        
        explanations = []
        for i, (image, label) in enumerate(zip(images, labels)):
            explanation = self.explain(image, label, **kwargs)
            explanations.append(explanation)
        
        return explanations
    
    def get_explanation_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Create a standardized explanation dictionary.
        
        This helper method ensures all explainers return data in a consistent
        format. Subclasses can use this as a template and add their specific fields.
        
        Args:
            **kwargs: Key-value pairs to include in the explanation
        
        Returns:
            Standardized explanation dictionary
        
        Example:
            >>> result = self.get_explanation_dict(
            ...     explanation_type='attention',
            ...     prediction=2,
            ...     confidence=0.87,
            ...     attention_map=attn_map
            ... )
        
        Future Extensions:
            - Add validation to ensure required fields are present
            - Support for nested explanation structures
            - Automatic dtype/device conversion for consistency
        """
        return {
            'explanation_type': kwargs.get('explanation_type', 'unknown'),
            'prediction': kwargs.get('prediction', -1),
            'confidence': kwargs.get('confidence', 0.0),
            **{k: v for k, v in kwargs.items() 
               if k not in ['explanation_type', 'prediction', 'confidence']}
        }
    
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Handles common preprocessing steps like:
        - Adding batch dimension if needed
        - Moving to correct device
        - Ensuring correct dtype
        
        Args:
            image: Input image tensor
        
        Returns:
            Preprocessed image ready for model
        
        Note:
            Subclasses can override this if they need specific preprocessing
        """
        # Add batch dimension if needed
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        # Move to device and ensure float32
        image = image.to(self.device, dtype=torch.float32)
        
        return image
    
    def _postprocess_explanation(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess explanation before returning.
        
        Common postprocessing steps:
        - Convert tensors to numpy arrays
        - Normalize values to [0, 1] range
        - Move data to CPU
        
        Args:
            explanation: Raw explanation dictionary
        
        Returns:
            Postprocessed explanation dictionary
        
        Future Work:
            - Add options for keeping tensors vs converting to numpy
            - Support for different normalization strategies
            - Automatic dtype conversion based on output format
        """
        processed = {}
        for key, value in explanation.items():
            if isinstance(value, torch.Tensor):
                # Move to CPU and convert to numpy
                processed[key] = value.detach().cpu().numpy()
            else:
                processed[key] = value
        
        return processed
    
    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"{self.__class__.__name__}(device={self.device})"


"""
Future Explainer Types to Implement:

1. GradientExplainer(BaseExplainer):
   - Grad-CAM, Grad-CAM++
   - Integrated Gradients
   - SmoothGrad
   - Guided Backpropagation

2. PerturbationExplainer(BaseExplainer):
   - LIME (Local Interpretable Model-agnostic Explanations)
   - SHAP (SHapley Additive exPlanations)
   - Occlusion-based attribution

3. CounterfactualExplainer(BaseExplainer):
   - Generate minimal changes to flip prediction
   - Diversity in counterfactuals
   - Constrained optimization for realistic changes

4. UncertaintyExplainer(BaseExplainer):
   - Monte Carlo Dropout
   - Ensemble-based uncertainty
   - Diffusion-specific: variance across denoising runs

5. ConceptExplainer(BaseExplainer):
   - TCAV (Testing with Concept Activation Vectors)
   - Concept-based model explanations
   - Medical concept sensitivity analysis

Each of these will inherit from BaseExplainer and implement the explain() method
according to their specific technique, while maintaining the consistent interface.
"""

