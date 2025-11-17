"""
Inference Utilities for XAI Pipeline

This module provides functions for running inference and capturing intermediate states
needed for XAI analysis. Used when pre-computed predictions are insufficient.

Purpose:
- Generate predictions with intermediate states (patches, attns, y0_cond)
- Support batch inference for generating prediction CSVs
- Provide runtime inference when needed by XAI explainers
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import yaml
from easydict import EasyDict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffuser_trainer import CoolSystem
from xai.utils.image_utils import load_image, preprocess_for_model


def predict_with_intermediates(model: CoolSystem,
                               image_path: str,
                               image_size: int = 512,
                               normalize_mean: List[float] = [0.485, 0.456, 0.406],
                               normalize_std: List[float] = [0.229, 0.224, 0.225]) -> Dict[str, Any]:
    """
    Run inference on a single image and capture all intermediate states.
    
    Args:
        model: Loaded CoolSystem model
        image_path: Path to image file
        image_size: Target image size (default: 512)
        normalize_mean: Normalization mean (ImageNet default)
        normalize_std: Normalization std (ImageNet default)
    
    Returns:
        Dictionary containing:
        - 'prediction': Predicted class (int)
        - 'confidence': Confidence score (float)
        - 'probs': Class probabilities (np.array)
        - 'y0_aux': Auxiliary fusion prediction
        - 'y0_aux_global': Global pathway prediction
        - 'y0_aux_local': Local pathway prediction
        - 'patches': ROI patches tensor (bz, np, I, J)
        - 'attns': Patch attention weights (bz, np)
        - 'attn_map': Saliency map (bz, nc, H, W)
        - 'y0_cond': Guidance map M (bz, nc, np, np)
        - 'y_pred': Final diffusion prediction (bz, nc)
    """
    device = next(model.parameters()).device
    
    # Load and preprocess image
    image_pil = load_image(image_path, target_size=(image_size, image_size))
    image_tensor = preprocess_for_model(
        image_pil,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )
    image_tensor = image_tensor.to(device)
    
    # Run auxiliary model
    with torch.no_grad():
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = model.aux_model(image_tensor)
    
    # Prepare for diffusion
    bz, nc, H, W = attn_map.size()
    bz, np = attns.size()
    
    # Create guidance map
    y0_cond = model.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np)
    
    # Initialize with random noise
    yT = model.guided_prob_map(
        torch.rand_like(y0_aux_global),
        torch.rand_like(y0_aux_local),
        bz, nc, np
    )
    
    # Prepare attention
    attns_expanded = attns.unsqueeze(-1)
    attns_expanded = (attns_expanded * attns_expanded.transpose(1, 2)).unsqueeze(1)
    
    # Run diffusion
    with torch.no_grad():
        y_pred = model.DiffSampler.sample_high_res(
            image_tensor,
            yT,
            conditions=[y0_cond, patches, attns_expanded]
        )
    
    # Average over spatial dimensions
    y_pred = y_pred.reshape(bz, nc, np * np)
    y_pred = y_pred.mean(2)  # (bz, nc)
    
    # Get prediction
    probs = F.softmax(y_pred, dim=1)
    prediction = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, prediction].item())
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probs': probs[0].detach().cpu().numpy(),
        'y0_aux': y0_aux[0].detach().cpu().numpy(),
        'y0_aux_global': y0_aux_global[0].detach().cpu().numpy(),
        'y0_aux_local': y0_aux_local[0].detach().cpu().numpy(),
        'patches': patches.detach().cpu(),
        'attns': attns[0].detach().cpu().numpy(),
        'attn_map': attn_map.detach().cpu().numpy(),
        'y0_cond': y0_cond.detach().cpu().numpy(),
        'y_pred': y_pred[0].detach().cpu().numpy(),
        'image_path': str(image_path),
    }


def batch_predict(model: CoolSystem,
                 image_paths: List[str],
                 output_csv: str,
                 image_size: int = 512,
                 batch_size: int = 1,
                 normalize_mean: List[float] = [0.485, 0.456, 0.406],
                 normalize_std: List[float] = [0.229, 0.224, 0.225]) -> pd.DataFrame:
    """
    Run inference on a batch of images and save results to CSV.
    
    Args:
        model: Loaded CoolSystem model
        image_paths: List of image file paths
        output_csv: Path to save CSV file
        image_size: Target image size
        batch_size: Batch size for processing (default: 1)
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    
    Returns:
        DataFrame with predictions and probabilities
    """
    results = []
    
    device = next(model.parameters()).device
    num_classes = model.params.data.num_classes
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        for img_path in batch_paths:
            try:
                result = predict_with_intermediates(
                    model, img_path, image_size, normalize_mean, normalize_std
                )
                
                row = {
                    'image_path': result['image_path'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                }
                # Add class probabilities
                for cls in range(num_classes):
                    row[f'prob_{cls}'] = result['probs'][cls]
                
                results.append(row)
            except Exception as e:
                print(f"[inference] Error processing {img_path}: {e}")
                continue
        
        if (i + batch_size) % 10 == 0:
            print(f"[inference] Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[inference] Saved predictions to {output_csv}")
    
    return df

