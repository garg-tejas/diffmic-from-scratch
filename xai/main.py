"""
Main XAI Pipeline

This is the entry point for the XAI analysis framework.
It coordinates all components to generate comprehensive explanations.

Workflow:
1. Load configuration
2. Initialize model loader
3. Select representative samples
4. For each sample:
   - Run attention explainer
   - Run diffusion explainer
   - Generate visualizations
   - Create report
   - Save outputs
5. Generate summary report

Usage:
    python main.py --config config/xai_config.yaml
    python main.py --checkpoint logs/aptos/version_0/checkpoints/last.ckpt --samples 5
    python main.py --help
"""

import sys
import argparse
from pathlib import Path
import yaml
from easydict import EasyDict
from typing import Dict, Any, List
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import XAI components
from xai.core.model_loader import ModelLoader
from xai.core.sample_selector import SampleSelector
from xai.explainers.attention_explainer import AttentionExplainer
from xai.explainers.diffusion_explainer import DiffusionExplainer
from xai.explainers.guidance_explainer import GuidanceMapExplainer
from xai.explainers.feature_prior_explainer import FeaturePriorExplainer
from xai.explainers.noise_explainer import NoiseExplainer
from xai.explainers.faithfulness_validator import FaithfulnessValidator
from xai.explainers.conditional_attribution_explainer import ConditionalAttributionExplainer
from xai.explainers.spatiotemporal_trajectory_explainer import SpatioTemporalTrajectoryExplainer
from xai.explainers.counterfactual_explainer import GenerativeCounterfactualExplainer
from xai.visualizers.attention_vis import AttentionVisualizer
from xai.visualizers.trajectory_vis import TrajectoryVisualizer
from xai.visualizers.guidance_vis import GuidanceMapVisualizer
from xai.visualizers.feature_prior_vis import FeaturePriorVisualizer
from xai.visualizers.noise_vis import NoiseVisualizer
from xai.visualizers.faithfulness_vis import FaithfulnessVisualizer
from xai.visualizers.attribution_vis import AttributionVisualizer
from xai.visualizers.spatiotemporal_vis import SpatioTemporalVisualizer
from xai.visualizers.counterfactual_vis import CounterfactualVisualizer
from xai.visualizers.report_generator import ReportGenerator
from xai.utils.image_utils import load_image, preprocess_for_model


class XAIPipeline:
    """
    Main XAI analysis pipeline.
    
    This class orchestrates all components of the XAI framework to generate
    comprehensive explanations for a set of images.
    
    Attributes:
        config: Configuration dictionary
        model_loader: ModelLoader instance
        sample_selector: SampleSelector instance
        explainers: Dictionary of explainer instances
        visualizers: Dictionary of visualizer instances
        report_generator: ReportGenerator instance
        
    Usage:
        >>> pipeline = XAIPipeline('config/xai_config.yaml')
        >>> pipeline.run()
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the XAI pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert class_names integer keys to strings for EasyDict compatibility
        if 'data' in config_dict and 'class_names' in config_dict['data']:
            class_names = config_dict['data']['class_names']
            if isinstance(class_names, dict):
                # Convert integer keys to strings
                config_dict['data']['class_names'] = {str(k): v for k, v in class_names.items()}
        
        # Convert balanced_ratio list if present (EasyDict handles lists fine, but ensure it's a list)
        if 'sampling' in config_dict and 'balanced_ratio' in config_dict['sampling']:
            if isinstance(config_dict['sampling']['balanced_ratio'], list):
                # Keep as list, EasyDict will handle it
                pass
        
        self.config = EasyDict(config_dict)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("="*80)
        self.logger.info("XAI PIPELINE INITIALIZATION")
        self.logger.info("="*80)
        self.logger.info(f"Config loaded from: {config_path}")
        
        # Initialize components
        self.model_loader = None
        self.model = None
        self.sample_selector = None
        self.explainers = {}
        self.visualizers = {}
        self.report_generator = None
        
        # Output paths
        self.output_dir = Path(__file__).parent / self.config.output.output_dir
        self.images_dir = self.output_dir / self.config.output.images_dir
        self.html_dir = self.output_dir / self.config.output.html_dir
        self.arrays_dir = self.output_dir / self.config.output.arrays_dir
        
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.config.logging.verbosity >= 1 else logging.WARNING
        
        # Create logger
        self.logger = logging.getLogger('XAI')
        self.logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if requested
        if self.config.logging.save_log:
            log_file = Path(__file__).parent / self.config.logging.log_file
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def initialize_model(self):
        """Load the trained model from checkpoint."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("LOADING MODEL")
        self.logger.info("-"*80)
        
        # Construct paths (relative to project root)
        project_root = Path(__file__).parent.parent
        checkpoint_path = self.config.model.checkpoint_path
        config_path = self.config.model.config_path
        
        # Initialize model loader
        self.model_loader = ModelLoader(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=self.config.model.device
        )
        
        # Load model
        self.model = self.model_loader.load_model()
        
        # Print model info
        info = self.model_loader.get_model_info()
        self.logger.info(f"Model info:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def initialize_sample_selector(self):
        """Initialize sample selector and select samples."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("SELECTING SAMPLES")
        self.logger.info("-"*80)
        
        self.sample_selector = SampleSelector(
            csv_path=self.config.sampling.predictions_csv,
            num_classes=self.config.data.num_classes,
            random_seed=self.config.sampling.random_seed
        )
        
        # Select samples
        self.selected_samples = self.sample_selector.select_samples(
            samples_per_class=self.config.sampling.samples_per_class,
            strategy=self.config.sampling.selection_strategy,
            balanced_ratio=self.config.sampling.balanced_ratio
        )
        
        self.logger.info(f"Selected {len(self.selected_samples)} samples for analysis")
    
    def initialize_explainers(self):
        """Initialize explainer modules."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("INITIALIZING EXPLAINERS")
        self.logger.info("-"*80)
        
        device = self.model_loader.device
        explainer_config = {'num_classes': self.config.data.num_classes}
        
        # Attention explainer
        if self.config.explainers.attention.enabled:
            self.explainers['attention'] = AttentionExplainer(
                model=self.model,
                device=device,
                config=explainer_config
            )
            self.logger.info("[OK] Attention explainer initialized")
        
        # Diffusion explainer
        if self.config.explainers.diffusion.enabled:
            self.explainers['diffusion'] = DiffusionExplainer(
                model=self.model,
                device=device,
                config={**explainer_config, 'subsample_timesteps': self.config.explainers.diffusion.num_timesteps_to_visualize}
            )
            self.logger.info("[OK] Diffusion explainer initialized")
        
        # Guidance map explainer
        if self.config.explainers.get('guidance_map', {}).get('enabled', False):
            self.explainers['guidance_map'] = GuidanceMapExplainer(
                model=self.model,
                device=device,
                config=explainer_config
            )
            self.logger.info("[OK] Guidance map explainer initialized")
        
        # Feature prior explainer
        if self.config.explainers.get('feature_prior', {}).get('enabled', False):
            self.explainers['feature_prior'] = FeaturePriorExplainer(
                model=self.model,
                device=device,
                config=explainer_config
            )
            self.logger.info("[OK] Feature prior explainer initialized")
        
        # Noise explainer
        if self.config.explainers.get('noise_analysis', {}).get('enabled', False):
            self.explainers['noise'] = NoiseExplainer(
                model=self.model,
                device=device,
                config=explainer_config
            )
            self.logger.info("[OK] Noise explainer initialized")
        
        # Faithfulness validator
        if self.config.explainers.get('faithfulness', {}).get('enabled', False):
            faithfulness_config = {
                **explainer_config,
                'deletion_steps': self.config.explainers.faithfulness.get('deletion_steps', 20),
                'occlusion_method': self.config.explainers.faithfulness.get('occlusion_method', 'blur'),
                'augmentation_count': self.config.explainers.faithfulness.get('augmentation_count', 10),
            }
            self.explainers['faithfulness'] = FaithfulnessValidator(
                model=self.model,
                device=device,
                config=faithfulness_config
            )
            self.logger.info("[OK] Faithfulness validator initialized")
        
        # Conditional attribution explainer
        if self.config.explainers.get('conditional_attribution', {}).get('enabled', False):
            self.explainers['conditional_attribution'] = ConditionalAttributionExplainer(
                model=self.model,
                device=device,
                config=explainer_config
            )
            self.logger.info("[OK] Conditional attribution explainer initialized")
        
        # Spatio-temporal trajectory explainer
        if self.config.explainers.get('spatiotemporal', {}).get('enabled', False):
            spatiotemporal_config = {
                **explainer_config,
                'track_timesteps': self.config.explainers.spatiotemporal.get('track_timesteps', [900, 700, 500, 300, 100, 10]),
            }
            self.explainers['spatiotemporal'] = SpatioTemporalTrajectoryExplainer(
                model=self.model,
                device=device,
                config=spatiotemporal_config
            )
            self.logger.info("[OK] Spatio-temporal trajectory explainer initialized")
        
        # Counterfactual explainer
        if self.config.explainers.get('counterfactual', {}).get('enabled', False):
            counterfactual_config = {
                **explainer_config,
                'guidance_scale': self.config.explainers.counterfactual.get('guidance_scale', 5.0),
            }
            self.explainers['counterfactual'] = GenerativeCounterfactualExplainer(
                model=self.model,
                device=device,
                config=counterfactual_config
            )
            self.logger.info("[OK] Counterfactual explainer initialized")
    
    def initialize_visualizers(self):
        """Initialize visualization modules."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("INITIALIZING VISUALIZERS")
        self.logger.info("-"*80)
        
        vis_config = {
            'colormap': self.config.visualization.colormap,
            'overlay_alpha': self.config.visualization.overlay_alpha,
            'figure_size': self.config.visualization.figure_size,
            'image_dpi': self.config.output.image_dpi,
            'class_names': self.config.data.class_names,
            'save_pdf': self.config.output.get('save_pdf', True),
        }
        
        self.visualizers['attention'] = AttentionVisualizer(vis_config)
        self.visualizers['trajectory'] = TrajectoryVisualizer(vis_config)
        self.visualizers['guidance_map'] = GuidanceMapVisualizer(vis_config)
        self.visualizers['feature_prior'] = FeaturePriorVisualizer(vis_config)
        self.visualizers['noise'] = NoiseVisualizer(vis_config)
        self.visualizers['faithfulness'] = FaithfulnessVisualizer(vis_config)
        self.visualizers['attribution'] = AttributionVisualizer(vis_config)
        self.visualizers['spatiotemporal'] = SpatioTemporalVisualizer(vis_config)
        self.visualizers['counterfactual'] = CounterfactualVisualizer(vis_config)
        if self.config.output.get('save_html', False):
            self.report_generator = ReportGenerator(vis_config)
        else:
            self.report_generator = None
        
        self.logger.info("[OK] All visualizers initialized")

    
    def _reset_cuda_context(self):
        """Reset CUDA context if it's corrupted."""
        try:
            import torch
            if torch.cuda.is_available():
                # Force synchronization and clear cache
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Reset the error state - get device index from device object
                device = self.model_loader.device
                if device.type == 'cuda':
                    device_idx = device.index if device.index is not None else 0
                    torch.cuda.set_device(device_idx)
        except Exception as e:
            self.logger.warning(f"CUDA reset warning: {e}")
    
    def process_sample(self, sample_row, sample_idx: int) -> Dict[str, Any]:
        """
        Process a single sample through the XAI pipeline.
        
        Args:
            sample_row: Row from selected samples DataFrame
            sample_idx: Index of this sample
        
        Returns:
            Dictionary with all results and paths
        """
        # Get sample info
        img_path_rel = sample_row['image_path']
        img_path = Path(__file__).parent.parent / img_path_rel
        ground_truth = int(sample_row['true_label'])
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing sample {sample_idx + 1}/{len(self.selected_samples)}")
        self.logger.info(f"Image: {img_path.name}")
        # Convert ground_truth to string for class_names lookup (keys are strings after EasyDict conversion)
        gt_str = str(ground_truth) if ground_truth != -1 else -1
        gt_name = self.config.data.class_names.get(gt_str, f"Class {ground_truth}") if gt_str != -1 else "Unknown"
        self.logger.info(f"Ground truth: {gt_name}")
        self.logger.info(f"{'='*60}")
        
        # Load and preprocess image
        try:
            image_pil = load_image(
                img_path,
                target_size=(self.config.data.image_size, self.config.data.image_size)
            )
            image_tensor = preprocess_for_model(
                image_pil,
                normalize_mean=self.config.data.normalize_mean,
                normalize_std=self.config.data.normalize_std
            )
            
            # Move to device with error handling
            try:
                image_tensor = image_tensor.to(self.model_loader.device)
                # Synchronize to catch any async errors
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.logger.warning(f"CUDA error detected, attempting to reset context: {e}")
                    self._reset_cuda_context()
                    # Retry once after reset
                    try:
                        image_tensor = image_tensor.to(self.model_loader.device)
                        torch.cuda.synchronize()
                    except RuntimeError as e2:
                        self.logger.error(f"CUDA error persists after reset: {e2}")
                        raise
                else:
                    raise
        except Exception as e:
            self.logger.error(f"Failed to load/preprocess image: {e}")
            raise
        
        results = {
            'image_path': str(img_path_rel),
            'ground_truth': ground_truth,
            'sample_idx': sample_idx
        }
        
        visualizations = {}
        
        # Run attention explainer
        if 'attention' in self.explainers:
            try:
                self.logger.info("Running attention explainer...")
                attention_exp = self.explainers['attention'].explain(image_tensor, label=ground_truth)
                results['attention_exp'] = attention_exp
                results['attention_prediction'] = attention_exp['prediction']
                results['attention_confidence'] = attention_exp['confidence']
                results['attention_correct'] = (attention_exp['prediction'] == ground_truth)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.logger.warning(f"CUDA error in attention explainer: {e}")
                    self._reset_cuda_context()
                    raise
                raise
            
            # Generate visualizations
            import numpy as np
            image_array = np.array(image_pil)
            vis = self.visualizers['attention'].visualize_attention(
                image_array, attention_exp,
                save_path=self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attention.png"
            )
            visualizations['attention_overview'] = vis
            
            # Create interactive visualization if enabled
            if self.config.output.create_interactive:
                try:
                    interactive_path = self.html_dir / f"class_{ground_truth}_sample_{sample_idx}_interactive.html"
                    self.visualizers['attention'].create_interactive_visualization(
                        image_array, attention_exp,
                        save_path=interactive_path
                    )
                    results['interactive_attention_path'] = str(interactive_path.relative_to(self.output_dir))
                except Exception as e:
                    self.logger.warning(f"Failed to create interactive visualization: {e}")
        
        # Run faithfulness validator (after attention explainer)
        if 'faithfulness' in self.explainers and 'attention' in self.explainers:
            try:
                self.logger.info("Running faithfulness validator...")

                # Extract saliency map from attention explainer
                saliency_map = attention_exp.get('saliency_map')
                if saliency_map is not None:
                    faithfulness_exp = self.explainers['faithfulness'].explain(
                        image_tensor, 
                        label=ground_truth,
                        saliency_map=saliency_map
                    )
    
                    results['faithfulness_exp'] = faithfulness_exp
                    
                    # Generate visualizations
                    faithfulness_vis = self.visualizers['faithfulness']
                    
                    # Insertion/deletion curves
                    curves_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_faithfulness_curves.png"
                    faithfulness_vis.plot_insertion_deletion_curves(
                        faithfulness_exp['deletion_curve'],
                        faithfulness_exp['insertion_curve'],
                        faithfulness_exp['deletion_auc'],
                        faithfulness_exp['insertion_auc'],
                        save_path=curves_path
                    )
                    visualizations['faithfulness_curves'] = curves_path.relative_to(self.output_dir)
                    
                    # Summary plot
                    summary_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_faithfulness_summary.png"
                    faithfulness_vis.plot_faithfulness_summary(
                        faithfulness_exp,
                        save_path=summary_path
                    )
                    visualizations['faithfulness_summary'] = summary_path.relative_to(self.output_dir)
            except Exception as e:
                self.logger.exception("Failed to run faithfulness validator")
        
        # Run diffusion explainer
        if 'diffusion' in self.explainers:
            try:
                self.logger.info("Running diffusion explainer...")

                diffusion_exp = self.explainers['diffusion'].explain(image_tensor, label=ground_truth)

                results['diffusion_exp'] = diffusion_exp
                results['diffusion_prediction'] = diffusion_exp['prediction']
                results['diffusion_confidence'] = diffusion_exp['confidence']
                results['diffusion_correct'] = (diffusion_exp['prediction'] == ground_truth)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.logger.warning(f"CUDA error in diffusion explainer: {e}")
                    self._reset_cuda_context()
                    raise
                raise
            
            # Generate visualizations
            vis = self.visualizers['trajectory'].visualize_trajectory(
                diffusion_exp,
                save_path=self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_trajectory.png"
            )
            visualizations['trajectory'] = vis
            
            # Create denoising animation if enabled
            if self.config.output.create_animations:
                try:
                    animation_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_denoising_frames.png"
                    self.visualizers['trajectory'].create_denoising_animation(
                        diffusion_exp,
                        save_path=animation_path,
                        fps=self.config.output.animation_fps
                    )
                    results['denoising_animation_path'] = str(animation_path.relative_to(self.output_dir))
                except Exception as e:
                    self.logger.warning(f"Failed to create denoising animation: {e}")
            
            # Create attention evolution animation if diffusion explainer tracks attention
            if self.config.output.create_animations and 'attention_evolution' in diffusion_exp:
                try:
                    import numpy as np
                    image_array = np.array(image_pil)
                    attn_evolution_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attention_evolution_frames.png"
                    self.visualizers['attention'].create_attention_evolution_animation(
                        image_array,
                        diffusion_exp['attention_evolution'],
                        save_path=attn_evolution_path,
                        fps=self.config.output.animation_fps
                    )
                    results['attention_evolution_path'] = str(attn_evolution_path.relative_to(self.output_dir))
                except Exception as e:
                    self.logger.warning(f"Failed to create attention evolution animation: {e}")
            
            # Create combined animations if enabled
            if self.config.output.get('create_combined_animations', True):
                try:
                    from xai.visualizers.combined_visualizer import CombinedVisualizer
                    import numpy as np
                    image_array = np.array(image_pil)
                    
                    # Initialize combined visualizer
                    combined_vis = CombinedVisualizer(self.config.visualization)
                    
                    # Synchronized multi-panel animation
                    sync_anim_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_synchronized_frames.png"
                    combined_vis.create_synchronized_animation(
                        image_array, diffusion_exp, attention_exp,
                        save_path=sync_anim_path,
                        fps=self.config.output.animation_fps
                    )
                    results['synchronized_animation_path'] = str(sync_anim_path.relative_to(self.output_dir))
                    
                    # Image denoising sequence
                    denoising_seq_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_denoising_sequence_frames.png"
                    combined_vis.create_image_denoising_sequence(
                        image_array, diffusion_exp,
                        save_path=denoising_seq_path,
                        fps=self.config.output.animation_fps
                    )
                    results['denoising_sequence_path'] = str(denoising_seq_path.relative_to(self.output_dir))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create combined animations: {e}")
        
        # Run conditional attribution explainer (after diffusion, requires trajectory)
        if 'conditional_attribution' in self.explainers:
            try:
                self.logger.info("Running conditional attribution explainer...")

                attribution_exp = self.explainers['conditional_attribution'].explain(
                    image_tensor, 
                    label=ground_truth
                )

                results['attribution_exp'] = attribution_exp
                
                # Generate visualizations
                attribution_vis = self.visualizers['attribution']
                
                # Feature attribution bars
                bars_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attribution_bars.png"
                attribution_vis.plot_feature_attribution_bars(
                    attribution_exp['global_contribution'],
                    attribution_exp['roi_contribution_scores'],
                    save_path=bars_path
                )
                visualizations['attribution_bars'] = bars_path.relative_to(self.output_dir)
                
                # Guidance map heatmap
                heatmap_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_guidance_attribution.png"
                attribution_vis.plot_guidance_attribution_heatmap(
                    attribution_exp['guidance_map_attribution'],
                    save_path=heatmap_path
                )
                visualizations['guidance_attribution'] = heatmap_path.relative_to(self.output_dir)
                
                # Comprehensive comparison
                comparison_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attribution_comparison.png"
                attribution_vis.create_attribution_comparison(
                    ground_truth,
                    attribution_exp['prediction'],
                    attribution_exp,
                    save_path=comparison_path
                )
                visualizations['attribution_comparison'] = comparison_path.relative_to(self.output_dir)
            except Exception as e:
                self.logger.exception("Failed to run conditional attribution explainer")
        
        # Run spatio-temporal trajectory explainer
        if 'spatiotemporal' in self.explainers:
            try:
                self.logger.info("Running spatio-temporal trajectory explainer...")

                spatiotemporal_exp = self.explainers['spatiotemporal'].explain(
                    image_tensor,
                    label=ground_truth
                )

                results['spatiotemporal_exp'] = spatiotemporal_exp
                
                # Generate visualizations
                spatiotemporal_vis = self.visualizers['spatiotemporal']
                
                # Attention evolution plot
                evolution_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attention_evolution.png"
                spatiotemporal_vis.plot_attention_evolution(
                    spatiotemporal_exp['attention_trajectory'],
                    save_path=evolution_path
                )
                visualizations['attention_evolution'] = evolution_path.relative_to(self.output_dir)
                
                # Coarse-to-fine transition
                transition_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_coarse_to_fine.png"
                spatiotemporal_vis.plot_coarse_to_fine_transition(
                    spatiotemporal_exp['attention_trajectory'],
                    save_path=transition_path
                )
                visualizations['coarse_to_fine'] = transition_path.relative_to(self.output_dir)
                
                # Animated heatmap if enabled
                if self.config.output.create_animations:
                    try:
                        heatmap_anim_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attention_heatmap_sequence.png"
                        spatiotemporal_vis.create_attention_heatmap_sequence(
                            spatiotemporal_exp['attention_trajectory'],
                            save_path=heatmap_anim_path,
                            fps=self.config.output.animation_fps
                        )
                        results['attention_heatmap_animation_path'] = str(heatmap_anim_path.relative_to(self.output_dir))
                    except Exception as e:
                        self.logger.warning(f"Failed to create attention heatmap animation: {e}")
            except Exception as e:
                self.logger.exception("Failed to run spatio-temporal trajectory explainer")
        
        # Run guidance map explainer
        if 'guidance_map' in self.explainers:
            try:
                self.logger.info("Running guidance map explainer...")
                # Cleanup before explainer call

                try:
                    guidance_exp = self.explainers['guidance_map'].explain(image_tensor, label=ground_truth)
                    # Cleanup after explainer call
    
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.logger.warning(f"CUDA error in guidance map explainer: {e}")
                        self._reset_cuda_context()
                        raise
                    raise
                results['guidance_exp'] = guidance_exp
                
                # Generate visualizations
                guidance_vis = self.visualizers['guidance_map']
                
                # Guidance heatmap (all classes)
                guidance_heatmap_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_guidance_heatmap.png"
                guidance_vis.create_guidance_heatmap(
                    guidance_exp['guidance_map'],
                    save_path=guidance_heatmap_path
                )
                visualizations['guidance_heatmap'] = guidance_heatmap_path.relative_to(self.output_dir)
                
                # Interpolation plot
                interpolation_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_prior_interpolation.png"
                guidance_vis.create_interpolation_plot(
                    guidance_exp['global_prior'],
                    guidance_exp['local_prior'],
                    guidance_exp['distance_matrix'],
                    save_path=interpolation_path
                )
                visualizations['prior_interpolation'] = interpolation_path.relative_to(self.output_dir)
                
                # Prior comparison
                comparison_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_prior_comparison.png"
                guidance_vis.create_prior_comparison(
                    guidance_exp['global_prior'],
                    guidance_exp['local_prior'],
                    guidance_exp.get('fusion_prior'),
                    save_path=comparison_path
                )
                visualizations['prior_comparison'] = comparison_path.relative_to(self.output_dir)
                
                # Create guidance evolution animation from diffusion trajectory if available
                if self.config.output.create_animations and 'diffusion' in self.explainers:
                    try:
                        diffusion_exp = results.get('diffusion_exp', {})
                        if 'trajectory' in diffusion_exp:
                            guidance_anim_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_guidance_evolution.png"
                            guidance_vis.create_guidance_evolution_animation_from_trajectory(
                                diffusion_exp['trajectory'],
                                save_path=guidance_anim_path,
                                fps=1
                            )
                            results['guidance_evolution_path'] = str(guidance_anim_path.relative_to(self.output_dir))
                    except Exception as e:
                        self.logger.warning(f"Failed to create guidance evolution animation: {e}")
            except Exception as e:
                self.logger.exception("Failed to run guidance map explainer")
        
        # Run feature prior explainer
        if 'feature_prior' in self.explainers:
            try:
                self.logger.info("Running feature prior explainer...")
                # Cleanup before explainer call

                try:
                    feature_exp = self.explainers['feature_prior'].explain(image_tensor, label=ground_truth)
                    # Cleanup after explainer call
    
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.logger.warning(f"CUDA error in feature prior explainer: {e}")
                        self._reset_cuda_context()
                        raise
                    raise
                results['feature_exp'] = feature_exp
                
                # Generate visualizations
                feature_vis = self.visualizers['feature_prior']
                
                # Feature contribution plot
                contribution_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_feature_contributions.png"
                feature_vis.create_feature_contribution_plot(
                    feature_exp['contribution_scores'],
                    save_path=contribution_path
                )
                visualizations['feature_contributions'] = contribution_path.relative_to(self.output_dir)
                
                # Fusion weight heatmap
                fusion_weight_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_fusion_weights.png"
                feature_vis.create_fusion_weight_heatmap(
                    feature_exp['fusion_weights'],
                    save_path=fusion_weight_path
                )
                visualizations['fusion_weights'] = fusion_weight_path.relative_to(self.output_dir)
                
                # Feature space comparison (PCA or t-SNE)
                feature_space_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_feature_space.png"
                # Get method from config, default to 'pca' if not specified
                feature_method = self.config.explainers.get('feature_prior', {}).get('feature_space_method', 'pca')
                feature_vis.create_feature_comparison(
                    feature_exp['raw_features'],
                    feature_exp['roi_features'],
                    save_path=feature_space_path,
                    method=feature_method
                )
                visualizations['feature_space'] = feature_space_path.relative_to(self.output_dir)
                
                # ROI contribution bars
                roi_contribution_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_roi_importance.png"
                feature_vis.create_roi_contribution_bars(
                    feature_exp['roi_features'],
                    feature_exp.get('patch_attention'),
                    save_path=roi_contribution_path
                )
                visualizations['roi_importance'] = roi_contribution_path.relative_to(self.output_dir)
                
                # Create feature evolution animation if enabled
                if self.config.output.create_animations and 'diffusion' in self.explainers:
                    try:
                        # Create feature sequence from diffusion trajectory (simplified)
                        # In practice, this would track features at each timestep
                        diffusion_exp = results.get('diffusion_exp', {})
                        if 'trajectory' in diffusion_exp:
                            # Create simplified feature sequence from trajectory
                            feature_sequence = []
                            for step in diffusion_exp['trajectory']:
                                feature_sequence.append({
                                    'timestep': step.get('timestep', 0),
                                    'contribution_scores': feature_exp.get('contribution_scores', {}),
                                    'roi_features': feature_exp.get('roi_features'),
                                    'fusion_weights': feature_exp.get('fusion_weights'),
                                    'patch_attention': feature_exp.get('patch_attention'),  # Add patch attention for ROI visualization
                                    'prediction': step.get('predicted_class', feature_exp.get('prediction', 0)),
                                    'confidence': float(step.get('probs', [0.0] * 5)[step.get('predicted_class', 0)])
                                })
                            
                            feature_anim_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_feature_evolution.png"
                            feature_vis.create_feature_evolution_animation(
                                feature_sequence,
                                save_path=feature_anim_path,
                                fps=1
                            )
                            results['feature_evolution_path'] = str(feature_anim_path.relative_to(self.output_dir))
                    except Exception as e:
                        self.logger.warning(f"Failed to create feature evolution animation: {e}")
            except Exception as e:
                self.logger.exception("Failed to run feature prior explainer")
        
        # Run counterfactual explainer (for interesting cases)
        if 'counterfactual' in self.explainers:
            # Only generate counterfactuals for specific transitions if configured
            target_transitions = self.config.explainers.counterfactual.get('target_transitions', [])
            should_generate = False
            
            if target_transitions:
                # Check if this sample matches any target transition
                current_pred = results.get('diffusion_prediction') or results.get('attention_prediction')
                if current_pred is not None:
                    for orig, target in target_transitions:
                        if current_pred == orig:
                            should_generate = True
                            target_class = target
                            break
            else:
                # Generate for all samples
                should_generate = True
                current_pred = results.get('diffusion_prediction') or results.get('attention_prediction')
                if current_pred is not None and current_pred < 4:
                    target_class = current_pred + 1
                else:
                    should_generate = False
            
            if should_generate:
                try:
                    self.logger.info(f"Running counterfactual explainer (target: {target_class})...")
    
                    counterfactual_exp = self.explainers['counterfactual'].explain(
                        image_tensor,
                        label=ground_truth,
                        target_class=target_class
                    )
    
                    results['counterfactual_exp'] = counterfactual_exp
                    
                    # Generate visualizations
                    counterfactual_vis = self.visualizers['counterfactual']
                    
                    # Counterfactual comparison
                    import numpy as np
                    image_array = np.array(image_pil)
                    comparison_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_counterfactual.png"
                    counterfactual_vis.visualize_counterfactual(
                        image_array,
                        counterfactual_exp['counterfactual_image'],
                        counterfactual_exp['delta_map'],
                        save_path=comparison_path,
                        original_class=counterfactual_exp['original_prediction'],
                        counterfactual_class=counterfactual_exp['counterfactual_prediction']
                    )
                    visualizations['counterfactual_comparison'] = comparison_path.relative_to(self.output_dir)
                except Exception as e:
                    self.logger.exception("Failed to run counterfactual explainer")
        
        # Generate HTML report
        if self.config.output.save_html and self.report_generator:
            self.logger.info("Generating HTML report...")
            report_path = self.html_dir / f"class_{ground_truth}_sample_{sample_idx}_report.html"
            
            sample_info = {
                'image_path': img_path_rel,
                'ground_truth': ground_truth
            }
            
            # Collect animation paths
            animation_paths = {
                'denoising': results.get('denoising_animation_path'),
                'attention_evolution': results.get('attention_evolution_path'),
                'synchronized': results.get('synchronized_animation_path'),
                'denoising_sequence': results.get('denoising_sequence_path'),
                'guidance_evolution': results.get('guidance_evolution_path'),
                'feature_evolution': results.get('feature_evolution_path'),
                'noise_evolution': results.get('noise_evolution_path'),
            }
            # Remove None values
            animation_paths = {k: v for k, v in animation_paths.items() if v is not None}
            
            self.report_generator.create_sample_report(
                sample_info=sample_info,
                attention_exp=results.get('attention_exp', {}),
                diffusion_exp=results.get('diffusion_exp', {}),
                guidance_exp=results.get('guidance_exp', {}),
                feature_exp=results.get('feature_exp', {}),
                noise_exp=results.get('noise_exp', {}),
                faithfulness_exp=results.get('faithfulness_exp', {}),
                attribution_exp=results.get('attribution_exp', {}),
                spatiotemporal_exp=results.get('spatiotemporal_exp', {}),
                counterfactual_exp=results.get('counterfactual_exp', {}),
                visualizations=visualizations,
                animation_paths=animation_paths,
                save_path=report_path
            )
            results['report_path'] = str(report_path.relative_to(self.output_dir))
        
        # Save arrays if requested
        if self.config.output.save_arrays:
            import numpy as np
            array_path = self.arrays_dir / f"class_{ground_truth}_sample_{sample_idx}_data.npz"
            
            # Prepare spatiotemporal trajectory data
            spatiotemporal_exp = results.get('spatiotemporal_exp', {})
            spatiotemporal_trajectory = spatiotemporal_exp.get('attention_trajectory', [])
            
            # Extract trajectory arrays for saving
            spatiotemporal_data = {}
            if spatiotemporal_trajectory:
                spatiotemporal_data['spatiotemporal_timesteps'] = np.array([step['timestep'] for step in spatiotemporal_trajectory])
                spatiotemporal_data['spatiotemporal_global_attention'] = np.array([step['global_attention'] for step in spatiotemporal_trajectory])
                spatiotemporal_data['spatiotemporal_local_attention'] = np.array([step['local_attention'] for step in spatiotemporal_trajectory])
                spatiotemporal_data['spatiotemporal_roi_attention'] = np.array([step['roi_attention'] for step in spatiotemporal_trajectory])
                spatiotemporal_data['spatiotemporal_predictions'] = np.array([step['prediction'] for step in spatiotemporal_trajectory])
                spatiotemporal_data['spatiotemporal_confidence'] = np.array([step['confidence'] for step in spatiotemporal_trajectory])
                spatiotemporal_data['spatiotemporal_probs'] = np.array([step['probs'] for step in spatiotemporal_trajectory])
            
            np.savez(
                array_path,
                attention_saliency=results.get('attention_exp', {}).get('saliency_map'),
                diffusion_trajectory=[step['probs'] for step in results.get('diffusion_exp', {}).get('trajectory', [])],
                **spatiotemporal_data
            )
        
        # Save results as JSON for summary report generation
        if self.config.output.get('save_json', True):
            import json
            json_path = self.output_dir / f"class_{ground_truth}_sample_{sample_idx}_results.json"
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_results_to_json_serializable(results)
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
                        
        self.logger.info(f"[OK] Sample {sample_idx + 1} complete")
        
        return results
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]]):
        """Generate summary HTML report."""
        if not (self.config.output.save_html and self.report_generator):
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING SUMMARY REPORT")
        self.logger.info("="*80)
        
        summary_path = self.html_dir / "index.html"
        self.report_generator.create_summary_report(all_results, summary_path)
        
        self.logger.info(f"[OK] Summary report saved to {summary_path}")
    
    def print_summary_statistics(self, all_results: List[Dict[str, Any]]):
        """Print summary statistics."""
        self.logger.info("\n" + "="*80)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("="*80)
        
        total = len(all_results)
        
        # Accuracy
        att_correct = sum(1 for r in all_results if r.get('attention_correct', False))
        diff_correct = sum(1 for r in all_results if r.get('diffusion_correct', False))
        
        self.logger.info(f"Total samples analyzed: {total}")
        self.logger.info(f"Auxiliary model accuracy: {att_correct}/{total} ({att_correct/total*100:.1f}%)")
        self.logger.info(f"Diffusion model accuracy: {diff_correct}/{total} ({diff_correct/total*100:.1f}%)")
        
        # Average confidence
        import numpy as np
        avg_att_conf = np.mean([r.get('attention_confidence', 0) for r in all_results])
        avg_diff_conf = np.mean([r.get('diffusion_confidence', 0) for r in all_results])
        
        self.logger.info(f"Average auxiliary confidence: {avg_att_conf:.3f}")
        self.logger.info(f"Average diffusion confidence: {avg_diff_conf:.3f}")
        
        # Agreement
        agreement = sum(1 for r in all_results 
                       if r.get('attention_prediction') == r.get('diffusion_prediction'))
        self.logger.info(f"Model agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
    
    def _convert_results_to_json_serializable(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results dictionary to JSON-serializable format."""
        import numpy as np
        import torch
        
        def convert_value(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, np.integer):
                return int(v)
            elif isinstance(v, np.floating):
                return float(v)
            elif isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().tolist()
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            else:
                return v
        
        return convert_value(results)
    
    def run(self):
        """Run the complete XAI pipeline."""
        start_time = datetime.now()
        
        try:
            # Initialize all components
            self.initialize_model()
            self.initialize_sample_selector()
            self.initialize_explainers()
            self.initialize_visualizers()
            
            # Process all samples
            self.logger.info("\n" + "="*80)
            self.logger.info("PROCESSING SAMPLES")
            self.logger.info("="*80)
            
            all_results = []
            
            if self.config.logging.show_progress:
                iterator = tqdm(self.selected_samples.iterrows(), 
                              total=len(self.selected_samples),
                              desc="Processing samples")
            else:
                iterator = self.selected_samples.iterrows()
            
            for idx, row in iterator:
                try:
                    results = self.process_sample(row, idx)
                    all_results.append(results)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.logger.error(f"CUDA error on sample {idx + 1}, attempting recovery...")
                        self._reset_cuda_context()
                        # Try to continue with next sample
                        self.logger.warning(f"Skipping sample {idx + 1} due to CUDA error")
                        continue
                    else:
                        raise
            
            # Generate summary report
            if self.config.output.save_html and self.report_generator:
                self.generate_summary_report(all_results)
            
            # Print statistics
            self.print_summary_statistics(all_results)
            
            # Save configuration used
            config_save_path = self.output_dir / "run_config.yaml"
            with open(config_save_path, 'w') as f:
                yaml.dump(dict(self.config), f, default_flow_style=False)
            self.logger.info(f"\n[OK] Configuration saved to {config_save_path}")
            
        except Exception as e:
            self.logger.error(f"\n[ERROR] Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "="*80)
            self.logger.info(f"PIPELINE COMPLETE")
            self.logger.info(f"Duration: {duration:.1f} seconds")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='XAI Analysis Pipeline for Diffusion-Based Classification')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/xai_config.yaml',
        help='Path to configuration file (default: config/xai_config.yaml)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Override checkpoint path from config'
    )
    parser.add_argument(
        '--samples',
        type=int,
        help='Override number of samples per class'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Construct config path (relative to xai directory)
    config_path = Path(__file__).parent / args.config
    
    # Create pipeline
    pipeline = XAIPipeline(config_path)
    
    # Apply command-line overrides
    if args.checkpoint:
        pipeline.config.model.checkpoint_path = args.checkpoint
    if args.samples:
        pipeline.config.sampling.samples_per_class = args.samples
    if args.output_dir:
        pipeline.config.output.output_dir = args.output_dir
    
    # Run pipeline
    pipeline.run()


if __name__ == '__main__':
    main()

