"""
Sample Selector Module

This module handles the selection of representative samples for XAI analysis.
It reads prediction results from CSV and selects a balanced, diverse set of images.

Purpose:
- Load predictions from CSV file
- Select diverse samples covering all classes
- Balance correct vs incorrect predictions
- Enable reproducible sample selection

Selection Strategies:
- Balanced: Mix of correct and incorrect predictions per class
- Correct only: Only correctly classified samples
- Incorrect only: Only misclassified samples (for error analysis)
- Random: Random selection within each class

Future Extensions:
- Confidence-based selection (high/low confidence samples)
- Diversity-based selection (maximize feature diversity)
- Active learning integration (select most informative samples)
- Stratified sampling by difficulty
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random


class SampleSelector:
    """
    Select representative samples for XAI analysis from prediction results.
    
    This class provides intelligent sample selection to ensure we analyze
    a diverse, representative set of images that cover different scenarios:
    - All classes (balanced)
    - Correct and incorrect predictions
    - Different confidence levels (if available)
    
    Attributes:
        csv_path: Path to the predictions CSV file
        df: DataFrame containing all predictions
        num_classes: Number of classes in the dataset
        random_seed: Seed for reproducible selection
    
    Usage:
        >>> selector = SampleSelector('preds_epoch40.csv', num_classes=5)
        >>> samples = selector.select_samples(
        ...     samples_per_class=5,
        ...     strategy='balanced',
        ...     balanced_ratio=[3, 2]  # 3 correct, 2 incorrect
        ... )
        >>> print(f"Selected {len(samples)} samples")
    """
    
    def __init__(self, 
                 csv_path: str,
                 num_classes: int = 5,
                 random_seed: int = 42):
        """
        Initialize the sample selector.
        
        Args:
            csv_path: Path to CSV file with columns ['image_path', 'predicted_label', 'correct', 'true_label']
            num_classes: Number of classes in the dataset
            random_seed: Random seed for reproducibility
        
        Note:
            csv_path is relative to the xai/ directory
        """
        self.csv_path = Path(__file__).parent.parent / csv_path
        self.num_classes = num_classes
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load CSV
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        print(f"[SampleSelector] Loading predictions from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Validate CSV structure
        required_cols = ['image_path', 'predicted_label', 'correct', 'true_label']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics."""
        total = len(self.df)
        correct = self.df['correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        print(f"[SampleSelector] Dataset Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Correct predictions: {correct} ({accuracy*100:.2f}%)")
        print(f"  Incorrect predictions: {total - correct} ({(1-accuracy)*100:.2f}%)")
        print(f"\n  Per-class breakdown:")
        
        for class_id in range(self.num_classes):
            class_df = self.df[self.df['true_label'] == class_id]
            class_total = len(class_df)
            class_correct = class_df['correct'].sum()
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"    Class {class_id}: {class_total} samples, "
                  f"{class_correct} correct ({class_acc*100:.1f}%)")
    
    def select_samples(self,
                      samples_per_class: int = 5,
                      strategy: str = 'balanced',
                      balanced_ratio: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Select representative samples for XAI analysis.
        
        Args:
            samples_per_class: Number of samples to select per class
            strategy: Selection strategy ('balanced', 'correct_only', 'incorrect_only', 'random')
            balanced_ratio: For 'balanced' strategy, ratio of [correct, incorrect] samples
                          e.g., [3, 2] means try to get 3 correct and 2 incorrect per class
        
        Returns:
            DataFrame with selected samples, same structure as input CSV
        
        Strategy Details:
            - balanced: Mix of correct/incorrect based on balanced_ratio
            - correct_only: Only correctly classified samples
            - incorrect_only: Only misclassified samples
            - random: Random selection regardless of correctness
        
        Note:
            If a class doesn't have enough samples of a certain type,
            the selector will get as many as available and adjust the ratio.
        
        Future Enhancements:
            - Confidence-based selection (select high/low confidence cases)
            - Diversity-based selection using feature embeddings
            - Hard negative mining (most confusing samples)
            - Temporal selection (select from different training epochs)
        """
        print(f"\n[SampleSelector] Selecting samples...")
        print(f"  Strategy: {strategy}")
        print(f"  Samples per class: {samples_per_class}")
        
        if balanced_ratio is None:
            balanced_ratio = [samples_per_class // 2 + samples_per_class % 2, 
                            samples_per_class // 2]
        
        selected_samples = []
        
        for class_id in range(self.num_classes):
            # Get all samples for this class
            class_df = self.df[self.df['true_label'] == class_id]
            
            if len(class_df) == 0:
                print(f"  Warning: No samples found for class {class_id}")
                continue
            
            # Select based on strategy
            if strategy == 'balanced':
                class_samples = self._select_balanced(class_df, samples_per_class, balanced_ratio)
            elif strategy == 'correct_only':
                class_samples = self._select_correct_only(class_df, samples_per_class)
            elif strategy == 'incorrect_only':
                class_samples = self._select_incorrect_only(class_df, samples_per_class)
            elif strategy == 'random':
                class_samples = self._select_random(class_df, samples_per_class)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            selected_samples.append(class_samples)
            print(f"  Class {class_id}: Selected {len(class_samples)} samples "
                  f"({class_samples['correct'].sum()} correct, "
                  f"{len(class_samples) - class_samples['correct'].sum()} incorrect)")
        
        # Combine all selected samples
        result_df = pd.concat(selected_samples, ignore_index=True)
        
        print(f"\n[SampleSelector] [OK] Selected {len(result_df)} total samples")
        
        return result_df
    
    def _select_balanced(self, 
                        class_df: pd.DataFrame, 
                        n_samples: int,
                        ratio: List[int]) -> pd.DataFrame:
        """
        Select balanced mix of correct and incorrect predictions.
        
        Args:
            class_df: DataFrame for a single class
            n_samples: Total number of samples to select
            ratio: [n_correct, n_incorrect] desired
        
        Returns:
            DataFrame with selected samples
        """
        n_correct_desired, n_incorrect_desired = ratio
        
        # Get correct and incorrect samples
        correct_df = class_df[class_df['correct'] == 1]
        incorrect_df = class_df[class_df['correct'] == 0]
        
        # Adjust if not enough samples available
        n_correct_available = len(correct_df)
        n_incorrect_available = len(incorrect_df)
        
        n_correct = min(n_correct_desired, n_correct_available)
        n_incorrect = min(n_incorrect_desired, n_incorrect_available)
        
        # If we still need more samples, take from whichever pool has more
        total_selected = n_correct + n_incorrect
        if total_selected < n_samples:
            shortage = n_samples - total_selected
            if n_correct_available - n_correct > 0:
                n_correct += min(shortage, n_correct_available - n_correct)
            elif n_incorrect_available - n_incorrect > 0:
                n_incorrect += min(shortage, n_incorrect_available - n_incorrect)
        
        # Sample
        selected_correct = correct_df.sample(n=n_correct, random_state=self.random_seed) if n_correct > 0 else pd.DataFrame()
        selected_incorrect = incorrect_df.sample(n=n_incorrect, random_state=self.random_seed) if n_incorrect > 0 else pd.DataFrame()
        
        return pd.concat([selected_correct, selected_incorrect], ignore_index=True)
    
    def _select_correct_only(self, class_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Select only correctly classified samples."""
        correct_df = class_df[class_df['correct'] == 1]
        n = min(n_samples, len(correct_df))
        return correct_df.sample(n=n, random_state=self.random_seed)
    
    def _select_incorrect_only(self, class_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Select only misclassified samples."""
        incorrect_df = class_df[class_df['correct'] == 0]
        n = min(n_samples, len(incorrect_df))
        if n == 0:
            print(f"    Warning: No incorrect samples available")
            return pd.DataFrame()
        return incorrect_df.sample(n=n, random_state=self.random_seed)
    
    def _select_random(self, class_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Select random samples regardless of correctness."""
        n = min(n_samples, len(class_df))
        return class_df.sample(n=n, random_state=self.random_seed)
    
    def get_sample_info(self, sample_row: pd.Series) -> Dict[str, Any]:
        """
        Extract information from a sample row.
        
        Args:
            sample_row: A row from the selected samples DataFrame
        
        Returns:
            Dictionary with sample information
        
        Future:
            - Add confidence scores if available
            - Include feature embeddings
            - Add similarity to other samples
        """
        return {
            'image_path': sample_row['image_path'],
            'prediction': int(sample_row['predicted_label']),
            'ground_truth': int(sample_row['true_label']),
            'is_correct': bool(sample_row['correct']),
            'class_name': self._get_class_name(int(sample_row['true_label']))
        }
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Get human-readable class name.
        
        Args:
            class_id: Numeric class ID
        
        Returns:
            Class name string
        
        Note:
            These names are specific to APTOS dataset (diabetic retinopathy)
            For other datasets, this should be configurable
        """
        class_names = {
            0: "No DR",
            1: "Mild",
            2: "Moderate", 
            3: "Severe",
            4: "Proliferative DR"
        }
        # Handle both string and integer keys (after EasyDict conversion, keys may be strings)
        class_id_str = str(class_id)
        if isinstance(class_names, dict):
            # Try string key first (EasyDict converts int keys to strings)
            if class_id_str in class_names:
                return class_names[class_id_str]
            # Fallback to integer key
            elif class_id in class_names:
                return class_names[class_id]
        return f"Class {class_id}"
    
    def export_selected_samples(self, 
                               selected_df: pd.DataFrame, 
                               output_path: str):
        """
        Export selected samples to a new CSV file.
        
        Args:
            selected_df: DataFrame with selected samples
            output_path: Path to save the CSV
        
        This is useful for:
        - Keeping track of which samples were analyzed
        - Reproducibility
        - Sharing sample selections
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(output_path, index=False)
        print(f"[SampleSelector] Exported selected samples to {output_path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SampleSelector(csv={self.csv_path.name}, total_samples={len(self.df)}, classes={self.num_classes})"


"""
Usage Example:

# Basic usage
selector = SampleSelector('preds_epoch40.csv', num_classes=5, random_seed=42)

# Select 5 samples per class with balanced correct/incorrect ratio
samples = selector.select_samples(
    samples_per_class=5,
    strategy='balanced',
    balanced_ratio=[3, 2]  # 3 correct, 2 incorrect
)

# Get info for a specific sample
for idx, row in samples.iterrows():
    info = selector.get_sample_info(row)
    print(f"Sample: {info['image_path']}")
    print(f"  Predicted: {info['prediction']}, Ground truth: {info['ground_truth']}")
    print(f"  Correct: {info['is_correct']}")

# Export for reproducibility
selector.export_selected_samples(samples, 'outputs/selected_samples.csv')

Future Extensions:
1. Confidence-based selection:
   - Select high-confidence correct predictions (model is very sure and right)
   - Select high-confidence incorrect predictions (model is very sure but wrong)
   - Select low-confidence predictions (model is uncertain)

2. Feature diversity:
   - Use pretrained embeddings to measure diversity
   - Maximize coverage of feature space
   - Avoid redundant similar samples

3. Hard negative mining:
   - Select samples near decision boundaries
   - Select samples that confuse the model
   - Select samples with high loss values

4. Active learning integration:
   - Select most informative samples for human review
   - Prioritize samples that would improve model if retrue_labeled
   - Identify potential true_label errors in dataset
"""

