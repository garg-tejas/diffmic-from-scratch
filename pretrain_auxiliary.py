"""
Auxiliary Model Pretraining Script for DiffMIC-v2

Usage:
    python pretrain_auxiliary.py --config configs/aptos.yml --epochs 100 --batch_size 4 --use_amp
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
from easydict import EasyDict
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from pretraining.dcg import DCG
from utils import get_dataset, compute_aptos_metrics
from dataloader.loading import APTOSDataset


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = scaler is not None
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        with autocast('cuda', enabled=use_amp):
            y_fusion, y_global, y_local, _, _, _ = model(images)
            loss = criterion(y_fusion, labels) / accumulation_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        _, predicted = y_fusion.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device, epoch, use_amp=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            with autocast('cuda', enabled=use_amp):
                y_fusion, y_global, y_local, _, _, _ = model(images)
                loss = criterion(y_fusion, labels)
            
            total_loss += loss.item()
            all_preds.append(y_fusion)
            all_labels.append(nn.functional.one_hot(labels, num_classes=y_fusion.shape[1]).float())
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_aptos_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader)
    
    print(f"Val Metrics - Loss: {avg_loss:.4f}, Acc: {ACC:.4f}, F1: {F1:.4f}, "
          f"Precision: {Prec:.4f}, Recall: {Rec:.4f}, AUC: {AUC_ovo:.4f}, Kappa: {kappa:.4f}")
    
    return avg_loss, ACC, F1, AUC_ovo, kappa


def main():
    parser = argparse.ArgumentParser(description='Pretrain Auxiliary Model for DiffMIC-v2')
    parser.add_argument('--config', type=str, default='configs/aptos.yml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (reduced default for 512x512 images)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='pretraining/ckpt',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (FP16) for memory efficiency')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (simulates larger batch size)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)
    
    set_seed(config.data.seed)
    
    # Detect number of GPUs and set device
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPU(s) - Training will use DataParallel")
        device = torch.device('cuda:0')  # Use first GPU as primary
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading dataset...")
    preprocess_ben = getattr(config.data, "preprocess_ben", False)
    ben_sigma = getattr(config.data, "ben_sigma", 200)
    train_dataset = APTOSDataset(
        config.data.traindata,
        train=True,
        preprocess_ben=preprocess_ben,
        ben_sigma=ben_sigma
    )
    test_dataset = APTOSDataset(
        config.data.testdata,
        train=False,
        preprocess_ben=preprocess_ben,
        ben_sigma=ben_sigma
    )
    class_counts = [0.492, 0.101, 0.273, 0.053, 0.081]  # Your class percentages
    class_weights = [1 / c for c in class_counts]  # Inverse: ~[2.03, 9.90, 3.66, 18.87, 12.35]
    class_weights = np.array(class_weights) / np.sum(class_weights)  # Normalize to sum to 1

    sample_weights = [class_weights[label] for _, label in train_dataset]  # Assumes dataset returns (img, label)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(test_dataset)}")
    
    print("Initializing auxiliary model...")
    model = DCG(config).to(device)
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
        print(f"Effective batch size: {args.batch_size * num_gpus}")
    class_weights = torch.tensor([2.0325, 9.9010, 3.6630, 22.6415, 14.8148], dtype=torch.float).to(device)  # Inverse freq from above, adjust as needed
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    scaler = GradScaler('cuda') if args.use_amp else None
    if args.use_amp:
        print("Using mixed precision training (FP16)")
    if args.accumulation_steps > 1:
        print(f"Using gradient accumulation: {args.accumulation_steps} steps")
        print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint.get('best_f1', 0.0)
            print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    print("\n" + "="*50)
    print("Starting Auxiliary Model Pretraining")
    print("="*50 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1, 
                                            scaler=scaler, accumulation_steps=args.accumulation_steps)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        val_loss, val_acc, val_f1, val_auc, val_kappa = validate(model, val_loader, criterion, device, epoch + 1,
                                                                   use_amp=args.use_amp)
        
        scheduler.step()
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'best_f1': best_f1
        }
        
        last_path = os.path.join(args.output_dir, 'last_aux_model.pth')
        torch.save([checkpoint['model_state_dict']], last_path)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            checkpoint['best_f1'] = best_f1
            best_path = os.path.join(args.output_dir, 'aptos_aux_model.pth')
            torch.save([checkpoint['model_state_dict']], best_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s). Best F1: {best_f1:.4f}")
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered! No improvement for {args.patience} epochs.")
                print(f"Best F1: {best_f1:.4f}")
                break
        
        full_ckpt_path = os.path.join(args.output_dir, 'training_checkpoint.pth')
        torch.save(checkpoint, full_ckpt_path)
    
    print("\n" + "="*50)
    print("Pretraining Complete!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'aptos_aux_model.pth')}")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()