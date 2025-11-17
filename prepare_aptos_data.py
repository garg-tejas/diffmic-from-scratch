"""
APTOS 2019 Dataset Preparation Script
Converts CSV format to pickle format required by DiffMIC-v2

Creates train/test splits from train.csv only
The test set will be used as validation during training.

Usage:
    python prepare_aptos_data.py --csv_path /path/to/train.csv --img_dir /path/to/images --output_dir ./dataset
"""

import os
import argparse
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

def ben_preprocess(img_np, sigma=200):
    """
    Ben Graham preprocessing:
      - Contrast enhancement via unsharp masking
      - Circular mask to remove background/border
    Expects RGB np.ndarray, returns RGB np.ndarray.
    """
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        mask = np.zeros_like(img)
        cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
        img = cv2.bitwise_and(img, mask)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def preprocess_and_cache_image(src_path, dst_path, sigma=200, overwrite=False):
    """
    Preprocess a single image and cache it to disk.
    Returns the destination path (existing or newly created).
    """
    if not overwrite and os.path.exists(dst_path):
        return dst_path

    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at {src_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = ben_preprocess(img, sigma=sigma)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    Image.fromarray(processed).save(dst_path, format="PNG")
    return dst_path


def compute_gaussian_params(height, width, sigma):
    max_hw = max(3, min(height, width))
    # Cap sigma so that kernel fits inside the image. Gaussian kernel spans roughly 6*sigma.
    max_sigma = max(0.5, (max_hw - 1) / 6.0)
    sigma = float(max(0.5, min(sigma, max_sigma)))
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel_size = max(3, min(kernel_size, max_hw))
    if kernel_size % 2 == 0:
        kernel_size = kernel_size - 1 if kernel_size > 3 else kernel_size + 1
    # Recompute sigma to match kernel window if we were clamped
    sigma = min(sigma, (kernel_size - 1) / 6.0)
    sigma = max(0.5, sigma)
    return kernel_size, sigma


def build_circular_mask(batch):
    device = batch.device
    b, _, h, w = batch.shape
    gray = 0.2989 * batch[:, 0] + 0.5870 * batch[:, 1] + 0.1140 * batch[:, 2]
    gray = torch.clamp(gray, min=0.0)
    weights_sum = gray.sum(dim=(1, 2), keepdim=True) + 1e-6

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing='ij'
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    cx = (gray * xx).sum(dim=(1, 2), keepdim=True) / weights_sum
    cy = (gray * yy).sum(dim=(1, 2), keepdim=True) / weights_sum

    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = torch.sqrt((gray * dist ** 2).sum(dim=(1, 2), keepdim=True) / weights_sum)
    radius = torch.clamp(radius * 1.8, min=0.5, max=1.5)

    mask = (dist <= radius).float()

    invalid = (weights_sum.squeeze(-1).squeeze(-1) <= 1e-4)
    if invalid.any():
        mask[invalid] = 1.0

    return mask.unsqueeze(1)


def ben_preprocess_gpu_batch(batch, sigma):
    b, _, h, w = batch.shape
    kernel_size, sigma_eff = compute_gaussian_params(h, w, sigma)
    blurred = TF.gaussian_blur(batch, [kernel_size, kernel_size], [sigma_eff, sigma_eff])
    sharpened = torch.clamp(4 * batch - 4 * blurred + 128.0 / 255.0, 0.0, 1.0)
    mask = build_circular_mask(batch)
    return sharpened * mask


def preprocess_images_gpu(records, save_dir, sigma=200, batch_size=8, overwrite=False):
    if torch is None or TF is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA preprocessing requested but torch/torchvision with CUDA support is not available.")

    device = torch.device('cuda')
    os.makedirs(save_dir, exist_ok=True)

    processed = []
    pending = []

    for rec in records:
        cached_path = os.path.join(save_dir, f"{rec['id']}.png")
        if not overwrite and os.path.exists(cached_path):
            processed.append({'img_root': cached_path, 'label': rec['label']})
        else:
            pending.append((rec, cached_path))

    def flush(batch_tensors, metas):
        if not batch_tensors:
            return
        stacked = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            processed_batch = ben_preprocess_gpu_batch(stacked, sigma)
        processed_batch = processed_batch.cpu()
        for tensor, meta in zip(processed_batch, metas):
            arr = (tensor.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            arr = np.transpose(arr, (1, 2, 0))
            Image.fromarray(arr).save(meta['dst'], format='PNG')
            processed.append({'img_root': meta['dst'], 'label': meta['label']})

    current_tensors = []
    current_meta = []
    current_shape = None

    for rec, dst in tqdm(pending, desc="GPU Ben preprocessing", unit="img"):
        try:
            img = Image.open(rec['raw_path']).convert('RGB')
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img.close()
        except Exception as exc:
            print(f"Warning: GPU load failed for {rec['id']} ({exc}); falling back to CPU.")
            try:
                processed_path = preprocess_and_cache_image(rec['raw_path'], dst, sigma=sigma, overwrite=True)
                processed.append({'img_root': processed_path, 'label': rec['label']})
            except Exception as cpu_exc:
                print(f"Warning: CPU fallback also failed for {rec['id']} ({cpu_exc}), skipping.")
            continue

        shape = tensor.shape[1:]
        if current_shape is None:
            current_shape = shape
        if shape != current_shape or len(current_tensors) >= batch_size:
            flush(current_tensors, current_meta)
            current_tensors = []
            current_meta = []
            current_shape = shape

        current_tensors.append(tensor)
        current_meta.append({'dst': dst, 'label': rec['label']})

    flush(current_tensors, current_meta)

    return processed


def prepare_aptos_data(csv_path, img_dir, output_dir, train_ratio=0.75, seed=42,
                       preprocess=True, preprocess_dir="aptos_preprocessed", ben_sigma=200,
                       overwrite_preprocess=False, preprocess_backend="auto", gpu_batch_size=8):
    """
    Prepare APTOS 2019 dataset from CSV format to pickle format.
    Creates stratified train/test splits from train.csv only.
    """
    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")
    
    data_list = []
    preprocess_root = os.path.join(output_dir, preprocess_dir)

    records = []
    scan_iter = tqdm(df.iterrows(), total=len(df), desc="Scanning images") if preprocess else df.iterrows()
    for idx, row in scan_iter:
        img_name = row['id_code']
        label = int(row['diagnosis'])
        
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(img_dir, f"{img_name}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            print(f"Warning: Image not found for {img_name}, skipping...")
            continue
            
        records.append({'id': img_name, 'label': label, 'raw_path': img_path})

    if preprocess and records:
        backend = preprocess_backend.lower()
        if backend == 'auto':
            backend = 'gpu' if torch is not None and torch.cuda.is_available() else 'cpu'
        if backend == 'gpu' and (torch is None or TF is None or not torch.cuda.is_available()):
            print("CUDA preprocessing requested but not available; falling back to CPU pipeline.")
            backend = 'cpu'

        if backend == 'gpu':
            try:
                data_list = preprocess_images_gpu(
                    records,
                    save_dir=preprocess_root,
                    sigma=ben_sigma,
                    batch_size=gpu_batch_size,
                    overwrite=overwrite_preprocess
                )
            except Exception as exc:
                print(f"GPU preprocessing failed ({exc}); falling back to CPU pipeline.")
                backend = 'cpu'

        if backend == 'cpu':
            data_list = []
            iterable = tqdm(records, desc="CPU Ben preprocessing", unit="img")
            for rec in iterable:
                cached_path = os.path.join(preprocess_root, f"{rec['id']}.png")
                try:
                    img_path = preprocess_and_cache_image(
                        src_path=rec['raw_path'],
                        dst_path=cached_path,
                        sigma=ben_sigma,
                        overwrite=overwrite_preprocess
                    )
                    data_list.append({'img_root': img_path, 'label': rec['label']})
                except Exception as exc:
                    print(f"Warning: preprocessing failed for {rec['id']} ({exc}), skipping.")
        else:
            print("Using GPU accelerated preprocessing.")
    else:
        data_list = [{'img_root': rec['raw_path'], 'label': rec['label']} for rec in records]
    
    print(f"Valid samples after filtering: {len(data_list)}")
    
    test_ratio = 1.0 - train_ratio
    print(f"\nSplit ratios - Train: {train_ratio:.1%}, Test/Val: {test_ratio:.1%}")
    
    labels = [d['label'] for d in data_list]
    train_list, test_list = train_test_split(
        data_list, 
        train_size=train_ratio, 
        random_state=seed,
        stratify=labels
    )
    
    print(f"\nDataset splits:")
    print(f"  Train samples: {len(train_list)} ({len(train_list)/len(data_list)*100:.1f}%)")
    print(f"  Test/Val samples: {len(test_list)} ({len(test_list)/len(data_list)*100:.1f}%)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_pkl_path = os.path.join(output_dir, 'aptos_train_list.pkl')
    test_pkl_path = os.path.join(output_dir, 'aptos_test_list.pkl')
    
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_list, f)
    print(f"\nSaved train data to {train_pkl_path}")
    
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test_list, f)
    print(f"Saved test/validation data to {test_pkl_path}")
    
    train_labels = [d['label'] for d in train_list]
    test_labels = [d['label'] for d in test_list]
    
    print("\nTrain class distribution:")
    for i in range(5):
        count = train_labels.count(i)
        print(f"  Class {i}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    print("\nTest/Val class distribution:")
    for i in range(5):
        count = test_labels.count(i)
        print(f"  Class {i}: {count} ({count/len(test_labels)*100:.1f}%)")
    
    print("\nData preparation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare APTOS 2019 dataset for DiffMIC-v2',
        epilog='Note: Creates train/test splits from train.csv only (test.csv is unlabeled). '
               'Test set will be used as validation during training.'
    )
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to train.csv file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing the training images')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory for pickle files and cached images (default: ./dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='Training set ratio (default: 0.75, remaining 0.25 for validation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_preprocess', action='store_true',
                        help='Skip offline Ben preprocessing (raw images will be referenced)')
    parser.add_argument('--preprocess_dir', type=str, default='aptos_preprocessed',
                        help='Relative folder to cache preprocessed images under output_dir')
    parser.add_argument('--ben_sigma', type=float, default=200,
                        help='Sigma value for Gaussian blur in Ben preprocessing (default: 200)')
    parser.add_argument('--overwrite_preprocess', action='store_true',
                        help='Force reprocessing of images even if cached files exist')
    parser.add_argument('--preprocess_backend', type=str, default='auto', choices=['auto', 'cpu', 'gpu'],
                        help='Backend to use for Ben preprocessing (auto picks GPU if available)')
    parser.add_argument('--gpu_batch_size', type=int, default=8,
                        help='Batch size for GPU preprocessing (images grouped by shape)')
    
    args = parser.parse_args()
    
    prepare_aptos_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        preprocess=not args.no_preprocess,
        preprocess_dir=args.preprocess_dir,
        ben_sigma=args.ben_sigma,
        overwrite_preprocess=args.overwrite_preprocess,
        preprocess_backend=args.preprocess_backend,
        gpu_batch_size=args.gpu_batch_size
    )


if __name__ == '__main__':
    main()

