import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import torchvision.transforms as T

from torch.utils.data import DataLoader, WeightedRandomSampler
import pipeline
import math
from pretraining.dcg import DCG as AuxCls
from model import *
from utils import *


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr

        
        config_path = r'option/diff_DDIM.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        config = EasyDict(params)
        self.diff_opt = config

        self.model = ConditionalModel(self.params, guidance=self.params.diffusion.include_guidance)
        self.aux_model = AuxCls(self.params)
        aux_ckpt_path = 'pretraining/ckpt/aptos_aux_model.pth'
        self.init_weight(ckpt_path=aux_ckpt_path)
        self.aux_model.eval()

        self.save_hyperparameters()
        
        self.gts = []
        self.preds = []

        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train'),
        )
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['num_test_timesteps'])
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def configure_optimizers(self):
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)
        return [optimizer], [scheduler]


    def init_weight(self,ckpt_path=None):
        if ckpt_path:
            checkpoint = torch.load(ckpt_path,map_location=self.device)[0]
            checkpoint_model = checkpoint
            state_dict = self.aux_model.state_dict()
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            # print(checkpoint_model.keys())
            state_dict.update(checkpoint_model)
            self.aux_model.load_state_dict(state_dict) 

    def diffusion_focal_loss(self, prior, targets, noise, noise_gt, gamma=3, alpha=10):
        probs = F.softmax(prior, dim=1)
        probs = (probs * targets).sum(dim=1)
        weights = 1 + alpha * (1 - probs) ** gamma

        class_indices = torch.argmax(targets, dim=1)
        class_boost = torch.tensor([2.0325, 9.9010, 3.6630, 30.0, 16.0], device=probs.device)
        weights = weights * class_boost[class_indices]

        # Expand weights to match noise shape [batch_size, nc, np, np] for broadcasting
        # weights: [batch_size] -> [batch_size, 1, 1, 1]
        while weights.dim() < noise.dim():
            weights = weights.unsqueeze(-1)
        
        loss = weights * (noise - noise_gt).square()
        return loss.mean()

    def guided_prob_map(self, y0_g, y0_l, bz, nc, np):
    
        distance_to_diag = torch.tensor([[abs(i-j)  for j in range(np)] for i in range(np)]).to(self.device)

        weight_g = 1 - distance_to_diag / (np-1)
        weight_l = distance_to_diag / (np-1)
        interpolated_value = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        diag_indices = torch.arange(np)
        map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                map[i,j,diag_indices,diag_indices] = y0_g[i,j]
                map[i,j, np-1, 0] = y0_l[i,j]
                map[i,j, 0, np-1] = y0_l[i,j]
        return map

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
        
        
        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1,np*np,-1).reshape(bz*np*np,nc)
        noise = torch.randn_like(y_map).to(self.device)
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz*np*np,), device=self.device).long()

        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        y_fusion = torch.cat([y0_cond, noisy_y],dim=1)

        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns)

        noise = noise.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        loss = self.diffusion_focal_loss(y0_aux,y_batch,noise_pred,noise)
        
        # Scale loss by accumulation steps (handled automatically by Lightning, but we log effective batch size)
        accumulation_steps = getattr(self.params.training, 'accumulate_grad_batches', 1)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/effective_batch_size", self.params.training.batch_size * accumulation_steps, prog_bar=False)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/learning_rate", current_lr, prog_bar=False, on_step=True, on_epoch=False)
        
        return {"loss": loss}

    def on_validation_epoch_end(self):
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_aptos_metrics(gt, pred)

        # Log scalar metrics
        self.log('val/accuracy', ACC, sync_dist=True, prog_bar=True)
        self.log('val/balanced_accuracy', BACC, sync_dist=True)
        self.log('val/f1', F1, sync_dist=True, prog_bar=True)
        self.log('val/precision', Prec, sync_dist=True)        
        self.log('val/recall', Rec, sync_dist=True)
        self.log('val/auc', AUC_ovo, sync_dist=True, prog_bar=True)
        self.log('val/kappa', kappa, sync_dist=True)
        
        # Log ROC curves and confusion matrix
        if self.logger is not None:
            try:
                gt_np = gt.cpu().detach().numpy()
                pred_np = pred.cpu().detach().numpy()
                num_classes = gt_np.shape[1]
                
                gt_class = np.argmax(gt_np, axis=1)
                pred_class = np.argmax(pred_np, axis=1)
                
                # Log ROC curves for each class
                for class_idx in range(num_classes):
                    fpr, tpr, _ = roc_curve(gt_np[:, class_idx], pred_np[:, class_idx])
                    roc_auc = auc(fpr, tpr)
                    
                    # Create ROC curve plot
                    fig = plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - Class {class_idx}')
                    plt.legend(loc="lower right")
                    
                    # Convert plot to image tensor for TensorBoard
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    img = Image.open(buf)
                    img_tensor = T.ToTensor()(img)
                    plt.close(fig)
                    buf.close()
                    
                    # Log to TensorBoard
                    self.logger.experiment.add_image(
                        f'roc_curve/class_{class_idx}',
                        img_tensor,
                        self.current_epoch
                    )
                
                # Log confusion matrix
                cm = confusion_matrix(gt_class, pred_class)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=[f'Class {i}' for i in range(num_classes)],
                           yticklabels=[f'Class {i}' for i in range(num_classes)])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                img_tensor = T.ToTensor()(img)
                plt.close(fig)
                buf.close()
                
                self.logger.experiment.add_image(
                    'confusion_matrix',
                    img_tensor,
                    self.current_epoch
                )
            except Exception as e:
                print(f"Warning: Could not log ROC curves or confusion matrix: {e}")
        
        self.gts = []
        self.preds = []
        print("Val: Accuracy {0}, Balanced Acc {1}, F1 score {2}, Precision {3}, Recall {4}, AUROC {5}, Cohen Kappa {6}".format(
            ACC, BACC, F1, Prec, Rec, AUC_ovo, kappa))


    def validation_step(self,batch,batch_idx):
        self.model.eval()
        self.aux_model.eval()

        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()

        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        yT = self.guided_prob_map(torch.rand_like(y0_aux_global),torch.rand_like(y0_aux_local),bz,nc,np)
        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        y_pred = self.DiffSampler.sample_high_res(x_batch,yT,conditions=[y0_cond, patches, attns])
        y_pred = y_pred.reshape(bz, nc, np*np)
        y_pred = y_pred.mean(2)
        self.preds.append(y_pred)
        self.gts.append(y_batch)
    
    def train_dataloader(self):
        class_counts = [0.492, 0.101, 0.273, 0.053, 0.081]  # Your class percentages
        class_weights = [1 / c for c in class_counts]  # Inverse: ~[2.03, 9.90, 3.66, 18.87, 12.35]
        class_weights = np.array(class_weights) / np.sum(class_weights)  # Normalize to sum to 1

        preprocess_ben = getattr(self.params.data, "preprocess_ben", False)
        ben_sigma = getattr(self.params.data, "ben_sigma", 200)

        # Assign sample weights based on labels in your train_list
        train_dataset = APTOSDataset(
            self.params.data.traindata,
            train=True,
            preprocess_ben=preprocess_ben,
            ben_sigma=ben_sigma
        )
        sample_weights = [class_weights[label] for _, label in train_dataset]  # Assumes dataset returns (img, label)

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            sampler=sampler,
            num_workers=self.params.data.num_workers,
            drop_last=True
        )
        return train_loader
    
    def val_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
            drop_last=True
        )
        return test_loader  


def main():
    RESUME = False
    resume_checkpoint_path = r'checkpoints/last.ckpt'
    if RESUME == False:
        resume_checkpoint_path =None

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    config_path = r'configs/aptos.yml'
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    model = CoolSystem(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/f1',
        filename='aptos-epoch{epoch:02d}-accuracy-{val/accuracy:.4f}-f1-{val/f1:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=1,
        mode = "max",
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir='./logs',
        name='aptos_experiment',
        log_graph=False,
        default_hp_metric=False
    )
    
    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")
    
    # For now, use single GPU to avoid DDP batch size issues
    # Can enable multi-GPU later with proper batch sizing
    strategy = "ddp" if num_gpus > 1 else "auto" 
    devices = num_gpus
    print(f"Using {devices} GPUs")
    
    # Get gradient accumulation steps from config (default to 1 if not specified)
    accumulate_grad_batches = getattr(config.training, 'accumulate_grad_batches', 4)
    effective_batch_size = config.training.batch_size * accumulate_grad_batches
    print(f"Batch size: {config.training.batch_size}, Accumulation steps: {accumulate_grad_batches}, Effective batch size: {effective_batch_size}")
    
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=devices,
        precision=32,
        logger=tb_logger,
        strategy=strategy,
        enable_progress_bar=True,
        log_every_n_steps=5,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[checkpoint_callback, lr_monitor_callback]
    ) 

    trainer.fit(model,ckpt_path=resume_checkpoint_path)
    
if __name__ == '__main__':
    main()