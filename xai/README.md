# XAI Framework for Diffusion-Based Classification

Explainable AI framework for diffusion-based medical image classifiers. This framework focuses on explaining the Denoising U-Net (the actual decision-maker) rather than just the auxiliary DCG model.

## Quick Start

Run the pipeline with default settings:

```bash
cd xai
python main.py
```

Override specific settings via command line:

```bash
python main.py --checkpoint checkpoints/model.ckpt --samples 5 --output-dir my_outputs
```

## Configuration

Edit `config/xai_config.yaml` to configure model paths, sample selection, and which explainers to run. The config file controls:

- Model checkpoint and config paths
- Number of samples per class and selection strategy
- Which explainers are enabled (attention, diffusion, faithfulness, attribution, etc.)
- Visualization settings and output formats

## XAI-v2 Modules

The framework includes four new explainability modules that focus on the Denoising U-Net:

**Faithfulness Validator**: Validates that saliency maps actually highlight important regions using insertion/deletion games and semantic robustness tests.

**Conditional Attribution Explainer**: Quantifies how much the U-Net relies on global context versus local lesions by computing gradients through the entire reverse diffusion process.

**Spatio-Temporal Trajectory Explainer**: Tracks how attention evolves from global to local features across timesteps, showing the coarse-to-fine reasoning process.

**Generative Counterfactual Explainer**: Generates counterfactual examples showing what minimal changes would flip the prediction, using guided diffusion.

Enable or disable these modules in the config file under the `explainers` section.

## Output Structure

Results are saved to the output directory (default: `outputs/`):

- `html/index.html` - Summary report across all samples
- `html/class_X_sample_Y_report.html` - Individual sample reports with interactive visualizations
- `images/` - All visualization images and animations
- `arrays/` - NumPy arrays with raw explanation data (if enabled)

## Runtime Inference

If you need to run inference during analysis instead of using pre-computed predictions, use the inference utilities:

```python
from xai.utils.inference import load_model_for_inference, predict_with_intermediates

model = load_model_for_inference('checkpoints/model.ckpt', 'configs/aptos.yml')
result = predict_with_intermediates(model, 'path/to/image.png')
```

This returns patches, attention weights, guidance maps, and predictions needed by the explainers.

## Summary Reports

After running the main pipeline, generate aggregated reports:

```bash
python generate_xai_v2_report.py --results-dir outputs --output-dir paper_figures
```

## Troubleshooting

If you encounter checkpoint errors, verify the path in the config file or use the `--checkpoint` argument. For CUDA out of memory issues, reduce the number of samples per class or disable some explainers. The pipeline handles CUDA memory cleanup automatically and saves results incrementally.
