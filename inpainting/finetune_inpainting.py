"""
Fine-tune CFM model specifically for inpainting tasks
This improves inpainting quality by training on masked images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

#Add paths so Python can see everything
sys.path.append(current_dir) 
sys.path.append(parent_dir)  

from models.unet import TinyUNet, count_parameters
from trainer.cfm_trainer import ConditionalFlowMatcher, sample_cfm
from cfm_inpainting import InpaintingMaskGenerator


class InpaintingFlowMatcher(ConditionalFlowMatcher):
    """Extended Flow Matcher with inpainting-specific training"""
    
    def __init__(self, sigma_min=1e-4, mask_prob=0.5):
        super().__init__(sigma_min)
        self.mask_prob = mask_prob  # Probability of using masked training
        self.mask_gen = InpaintingMaskGenerator()
    
    def compute_inpainting_loss(self, model, x1, y, cfg_dropout_prob=0.1):
        """
        Compute flow matching loss with inpainting conditioning
        
        This trains the model to handle masked inputs, which improves
        inpainting quality during inference.
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Decide whether to use masked training for this batch
        use_masking = torch.rand(1).item() < self.mask_prob
        
        # Sample noise and time
        x0 = self.sample_noise(x1.shape, device)
        t = self.sample_time(batch_size, device)
        
        # Classifier-free guidance: randomly drop class conditioning
        if cfg_dropout_prob > 0:
            mask_cfg = torch.rand(batch_size, device=device) < cfg_dropout_prob
            y = torch.where(mask_cfg, torch.full_like(y, model.num_classes), y)
        
        if use_masking:
            # Generate random masks for training
            # Use variety of mask types
            mask_type = torch.rand(1).item()
            if mask_type < 0.25:
                mask = self.mask_gen.random_bbox_mask(x1.shape, min_size=4, max_size=24).to(device)
            elif mask_type < 0.5:
                mask = self.mask_gen.center_mask(x1.shape, mask_size=torch.randint(8, 20, (1,)).item()).to(device)
            elif mask_type < 0.75:
                mask = self.mask_gen.random_irregular_mask(x1.shape, num_strokes=torch.randint(3, 8, (1,)).item()).to(device)
            else:
                mask = self.mask_gen.half_image_mask(x1.shape, direction=['left', 'right', 'top', 'bottom'][torch.randint(0, 4, (1,)).item()]).to(device)
            
            # Interpolate normally
            x_t = self.interpolate(x0, x1, t)
            
            # Apply mask: keep original in unmasked regions, use interpolation in masked
            # This simulates the inpainting inference process
            x_t = mask * x1 + (1 - mask) * x_t
            
            # Target velocity (full image)
            v_target = self.target_velocity(x0, x1)
            
            # Predict velocity
            v_pred = model(x_t, t, y)
            
            # Compute loss
            # Option 1: Loss on full image (model learns to predict full velocity)
            loss_full = F.mse_loss(v_pred, v_target)
            
            # Option 2: Emphasize masked regions (helps model focus on inpainting)
            loss_masked = F.mse_loss(v_pred * (1 - mask), v_target * (1 - mask))
            
            # Combine losses
            loss = 0.7 * loss_full + 0.3 * loss_masked
            
        else:
            # Standard CFM loss (no masking)
            x_t = self.interpolate(x0, x1, t)
            v_target = self.target_velocity(x0, x1)
            v_pred = model(x_t, t, y)
            loss = F.mse_loss(v_pred, v_target)
        
        return loss


def get_dataloaders(batch_size=32):
    """Prepare CIFAR-10 dataloaders"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    
    return train_loader


def train_epoch(model, optimizer, flow_matcher, train_loader, device, epoch, use_amp=True, cfg_dropout_prob=0.15, scheduler=None):
    """Train for one epoch with inpainting-aware loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss = flow_matcher.compute_inpainting_loss(model, images, labels, cfg_dropout_prob)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Update OneCycleLR per batch if provided
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        with torch.no_grad():
            total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': total_loss / num_batches})
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches


def finetune(args):
    """Fine-tune model for inpainting"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pre-trained model
    print(f"Loading pre-trained model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model architecture from checkpoint
    model_args = checkpoint.get('args', {})
    
    model = TinyUNet(
        in_channels=3,
        out_channels=3,
        base_channels=model_args.get('base_channels', 32),
        channel_mults=model_args.get('channel_mults', [1, 2, 2]),
        num_res_blocks=model_args.get('num_res_blocks', 2),
        attention_resolutions=model_args.get('attention_resolutions', [16]),
        num_classes=10,
        dropout=model_args.get('dropout', 0.1),
    )
    
    # Load pre-trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    num_params = count_parameters(model)
    print(f"✓ Model loaded with {num_params:,} parameters")
    print(f"  Pre-trained epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Pre-trained loss: {checkpoint.get('train_loss', '?'):.4f}")
    
    # Initialize inpainting-aware flow matcher
    flow_matcher = InpaintingFlowMatcher(mask_prob=args.mask_prob)
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    if args.lr_scheduler == 'cosine':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                # Warmup: linearly increase from 0 to 1
                return (epoch + 1) / args.warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
    elif args.lr_scheduler == 'linear':
        # Linear decay with warmup
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                return 1 - progress * (1 - args.lr_min / args.lr)
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
    elif args.lr_scheduler == 'step':
        # Step decay: reduce LR by factor every N epochs
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        
    elif args.lr_scheduler == 'exponential':
        # Exponential decay
        from torch.optim.lr_scheduler import ExponentialLR
        # Calculate decay rate to reach lr_min after args.epochs
        decay_rate = (args.lr_min / args.lr) ** (1 / args.epochs)
        scheduler = ExponentialLR(optimizer, gamma=decay_rate)
        
    elif args.lr_scheduler == 'plateau':
        # Reduce on plateau (based on loss)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                      threshold=1e-4, min_lr=args.lr_min)
        
    elif args.lr_scheduler == 'onecycle':
        # One cycle policy (fast convergence)
        from torch.optim.lr_scheduler import OneCycleLR
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                               epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                               pct_start=0.3, anneal_strategy='cos')
        
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")
    
    # Get dataloader
    train_loader = get_dataloaders(args.batch_size)
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print(f"Mask probability: {args.mask_prob}")
    print(f"Initial learning rate: {args.lr}")
    print(f"LR scheduler: {args.lr_scheduler}")
    print(f"Warmup epochs: {args.warmup_epochs}" if args.lr_scheduler in ['cosine', 'linear'] else "")
    print(f"CFG dropout: {args.cfg_dropout_prob}\n")
    
    for epoch in range(args.epochs):
        # Train (pass scheduler for OneCycleLR)
        train_loss = train_epoch(
            model, optimizer, flow_matcher, train_loader, 
            device, epoch, args.use_amp, args.cfg_dropout_prob,
            scheduler=scheduler if args.lr_scheduler == 'onecycle' else None
        )
        
        # Update learning rate
        if args.lr_scheduler == 'plateau':
            # ReduceLROnPlateau needs the loss
            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']
        elif args.lr_scheduler == 'onecycle':
            # OneCycleLR updates per batch, not per epoch
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, LR={current_lr:.6f}")
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': current_lr,
        })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or train_loss < best_loss:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'args': model_args,
                'finetuned_for_inpainting': True,  # Mark as fine-tuned
                'finetune_settings': {
                    'mask_prob': args.mask_prob,
                    'cfg_dropout_prob': args.cfg_dropout_prob,
                    'finetune_epochs': epoch + 1,
                }
            }
            
            # Save latest
            torch.save(checkpoint_data, output_dir / 'checkpoint_finetuned_latest.pt')
            
            # Save best
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(checkpoint_data, output_dir / 'checkpoint_finetuned_best.pt')
                print(f"  → Saved best fine-tuned model (loss={best_loss:.4f})")
            
            # Save periodic
            if (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint_data, output_dir / f'checkpoint_finetuned_epoch_{epoch}.pt')
        
        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            print("  Generating samples...")
            samples = sample_cfm(
                model, ConditionalFlowMatcher(), num_samples=64, num_classes=10,
                device=device, num_steps=50, cfg_scale=3.0
            )
            samples = (samples + 1) / 2
            torchvision.utils.save_image(
                samples, output_dir / f'samples_finetuned_epoch_{epoch}.png',
                nrow=8, normalize=False
            )
    
    # Save training history
    with open(output_dir / 'finetuning_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nFine-tuning completed! Best loss: {best_loss:.4f}")
    print(f"Fine-tuned checkpoints saved to: {output_dir}")
    print(f"\nUse the fine-tuned model:")
    print(f"  python demo_inpainting.py --checkpoint {output_dir}/checkpoint_finetuned_best.pt")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CFM for inpainting')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/cfm_finetuned',
                        help='Output directory for fine-tuned model')
    
    # Fine-tuning parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs (20-50 recommended)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial learning rate (lower than initial training)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # Learning rate scheduler options
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear', 'step', 'exponential', 'plateau', 'onecycle'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs (for cosine/linear schedulers)')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Step size for step scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Decay factor for step scheduler')
    
    # Inpainting-specific parameters
    parser.add_argument('--mask_prob', type=float, default=0.5,
                        help='Probability of using masked training (0.5 = 50%% of batches)')
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.15,
                        help='CFG dropout probability (higher than training for better guidance)')
    
    # Training settings
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=5,
                        help='Generate samples every N epochs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fine-tuning CFM for Inpainting")
    print("=" * 60)
    print(f"Pre-trained checkpoint: {args.checkpoint}")
    print(f"Fine-tuning epochs: {args.epochs}")
    print(f"Mask probability: {args.mask_prob}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    finetune(args)


if __name__ == '__main__':
    main()