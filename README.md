# DiffMIC-v2: Diffusion-Based Medical Image Classification

A from-scratch implementation of **DiffMIC-v2** (Dual-granularity Conditional Guidance) for Diabetic Retinopathy grading using diffusion models.

## Architecture Overview

The model combines two key systems for medical image classification:

### 1. DCG (Dual-granularity Conditional Guidance)

- **Global Pathway**: EfficientNet-B0 backbone with EfficientSAM encoder for holistic retinal feature extraction
- **Local Pathway**: ResNet-based encoder with gated attention focusing on 6 high-resolution ROI patches (73×73)
- **Attention Module**: Learns to weight contributions from global and local features

### 2. Conditional Diffusion Model

- **Denoising U-Net**: Iteratively refines noisy class representations using DDIM scheduling
- **Conditional Convolutions**: Time-step aware convolutions with learnable embeddings
- **Guided Probability Maps**: Interpolates global/local predictions for spatial conditioning

## Why I built this (The Implementation Challenge)

Reproducing **DiffMIC-v2** from scratch was a deep dive into the intersection of generative modeling and clinical diagnostics. Most medical classifiers rely on static CNNs, but DiffMIC uses the iterative nature of diffusion to "refine" a diagnosis.

**The Hurdles:**

- **Compute Efficiency**: Training a 1000-step diffusion model on undergraduate-grade hardware required heavy use of Gradient Accumulation and DDIM sampling to reduce inference time by 90% while maintaining accuracy.
- **Dual-Granularity**: Implementing the local pathway meant ensuring the gated attention mechanism correctly prioritized micro-lesions (like hemorrhages) without losing the global retinal context.
- **Result Reproduction**: Successfully hit **84.1% Accuracy** on the APTOS 2019 dataset, matching the original paper's performance benchmarks.

## Technical Details

| Component               | Implementation                        |
| ----------------------- | ------------------------------------- |
| **Framework**           | PyTorch Lightning                     |
| **Diffusion Scheduler** | DDIM (1000 train / 10 test timesteps) |
| **Image Size**          | 512×512                               |
| **Global Encoder**      | EfficientNet-B0 + EfficientSAM        |
| **Local Encoder**       | ResNet-18                             |
| **Feature Dim**         | 6144                                  |
| **Optimizer**           | Adam with Cosine Annealing LR         |
| **Loss**                | Diffusion Focal Loss (class-weighted) |

### Training Optimizations

- **Gradient Accumulation**: 4 steps (effective batch size: 64)
- **Mixed Precision**: Supported via PyTorch AMP
- **Weighted Sampling**: Class-balanced sampling for imbalanced DR grades
- **Multi-GPU**: DDP support for distributed training

### Preprocessing Pipeline

- **Ben Graham Enhancement**: Contrast enhancement + circular masking for retinal fundus images
- **Data Augmentation**: Random rotation (±30°), horizontal/vertical flips, color jitter, sharpness adjustment
- **Normalization**: ImageNet statistics

## Project Structure

```
├── model.py                 # Conditional diffusion model & encoders
├── pipeline.py              # DDIM sampling pipeline
├── diffuser_trainer.py      # PyTorch Lightning training module
├── utils.py                 # Optimizers, metrics, utilities
├── pretrain_auxiliary.py    # DCG pretraining script
├── prepare_aptos_data.py    # Dataset preparation with Ben Graham preprocessing
├── configs/
│   └── aptos.yml            # Training configuration
├── option/
│   └── diff_DDIM.yaml       # DDIM scheduler configuration
├── dataloader/
│   ├── loading.py           # APTOS dataset loader
│   └── transforms.py        # Custom augmentations
├── pretraining/
│   ├── dcg.py               # Dual-granularity Conditional Guidance module
│   ├── modules.py           # ResNet, Attention, Global/Local networks
│   └── tools.py             # Cropping & ROI utilities
└── EfficientSAM/            # EfficientSAM encoder (external dependency)
```

## Quick Start

### 1. Prepare Dataset

```bash
python prepare_aptos_data.py --csv_path /path/to/train.csv --img_dir /path/to/images --output_dir ./dataset
```

### 2. Pretrain Auxiliary Model (DCG)

```bash
python pretrain_auxiliary.py --config configs/aptos.yml --epochs 100 --batch_size 4 --use_amp
```

### 3. Train Diffusion Model

```bash
python diffuser_trainer.py
```

The model checkpoints and TensorBoard logs will be saved to `./checkpoints` and `./logs` respectively.

## Configuration

Key settings in `configs/aptos.yml`:

```yaml
data:
  num_classes: 5 # DR grades 0-4
  preprocess_ben: False # Enable Ben Graham preprocessing

model:
  arch: efficientnet_b0
  feature_dim: 6144
  num_k: 6 # Number of ROI patches

diffusion:
  timesteps: 1000
  test_timesteps: 100
  include_guidance: True

training:
  batch_size: 16
  accumulate_grad_batches: 4
  n_epochs: 50
```

## Metrics & Logging

The training pipeline logs:

- **Scalar Metrics**: Accuracy, Balanced Accuracy, F1, Precision, Recall, AUROC, Cohen's Kappa
- **Visualizations**: Per-class ROC curves, Confusion matrices
- **Monitoring**: Learning rate, effective batch size

All metrics are logged to TensorBoard for real-time monitoring.

## XAI Framework

This codebase serves as the foundation for our **Explainable AI Framework** for diffusion-based models.

**[XAI Framework Repository](https://github.com/garg-tejas/xai-diffusion-based-models)**

The XAI framework provides:

- Temporal explanations across diffusion timesteps
- Feature attribution analysis
- Attention visualization for ROI patches

---

> Based on the paper: _DiffMIC-v2: Medical Image Classification via Improved Diffusion Network_
