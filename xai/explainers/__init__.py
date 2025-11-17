"""
Explainer modules for different XAI techniques.

Current implementations:
- attention_explainer: Extract built-in attention mechanisms
- diffusion_explainer: Track diffusion denoising trajectory

Future extensions:
- gradient_explainer: Grad-CAM, Integrated Gradients, etc.
- perturbation_explainer: LIME, SHAP, etc.
- counterfactual_explainer: Generate minimal changes to flip predictions
- uncertainty_explainer: Ensemble-based uncertainty quantification
"""

from .attention_explainer import AttentionExplainer
from .diffusion_explainer import DiffusionExplainer
from .guidance_explainer import GuidanceMapExplainer
from .feature_prior_explainer import FeaturePriorExplainer
from .noise_explainer import NoiseExplainer

__all__ = ['AttentionExplainer', 'DiffusionExplainer', 'GuidanceMapExplainer', 'FeaturePriorExplainer', 'NoiseExplainer']

