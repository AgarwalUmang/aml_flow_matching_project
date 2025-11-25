"""
Main training script for MeanFlow Matching
Following paper arXiv:2505.13447

KEY FEATURES:
- Dual time variables (r, t) 
- JVP for MeanFlow Identity
- 1-step sampling (no ODE integration!)
- Adaptive loss weighting
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from pathlib import Path
import json

# Import custom modules
import sys
sys.path.append('.')
from models.unet_mean import TinyUNetMeanFlow, count_parameters
from trainer.mean_trainer import (
    MeanFlowMatcher, 
    MeanFlowTrainer, 
    sample_meanflow_1step
)


def get_dataloaders(batch_size=128, num_workers=2, cache_in_memory=True):
    """Prepare CIFAR-10 dataloaders with optional memory caching"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
    ])
    
    # Download datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Cache in memory for faster access
    if cache_in_memory:
        print("Caching dataset in memory...")
        train_data = []
        for i in range(len(train_dataset)):
            train_data.append(train_dataset[i])
        
        # Create wrapper
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        train_dataset = CachedDataset(train_data)
        print(f"Cached {len(train_dataset)} training samples")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0,  # Use 0 workers when cached
        pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, test_loader


def train(args):
    """Main training function"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model - NOTE: Using TinyUNetMeanFlow with dual time variables
    model = TinyUNetMeanFlow(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_classes=10,
        dropout=args.dropout,
    )
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params > 1.5e6:
        print(f"WARNING: Model has {num_params/1e6:.2f}M parameters (target: <1.5M)")
    
    # Initialize MeanFlow matcher and trainer
    flow_matcher = MeanFlowMatcher()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    
    trainer = MeanFlowTrainer(
        model=model,
        optimizer=optimizer,
        flow_matcher=flow_matcher,
        device=device,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(os.path.join(output_dir, 'checkpoint_latest.pt')):
        checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_latest.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    print(f"\n{'='*70}")
    print("STARTING MEANFLOW TRAINING")
    print(f"{'='*70}")
    print("KEY FEATURES:")
    print("  ✓ Model: u(z, r, t, y) - average velocity with dual time variables")
    print("  ✓ Loss: MeanFlow Identity with JVP")
    print("  ✓ Sampling: 1-step (no ODE integration!)")
    print(f"  ✓ Epochs: {args.epochs}")
    print(f"  ✓ Batch size: {args.batch_size}")
    print(f"  ✓ Gradient accumulation: {args.grad_accum_steps}")
    print(f"  ✓ Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"  ✓ CFG dropout: {args.cfg_dropout_prob}")
    print(f"  ✓ Ratio equal (r=t): {args.ratio_equal}")
    print(f"  ✓ Loss weighting p: {args.loss_p}")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = trainer.train_epoch(
            train_loader, epoch, 
            cfg_dropout_prob=args.cfg_dropout_prob,
            ratio_equal=args.ratio_equal,
            loss_p=args.loss_p
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, LR={current_lr:.6f}")
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': current_lr,
        })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or train_loss < best_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'args': vars(args),
                'model_type': 'meanflow',  # Mark this as true MeanFlow model
            }
            
            # Save latest
            torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
            
            # Save best
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
                print(f"  → Saved best model (loss={best_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            print(f"  Generating samples with 1-step sampling...")
            
            # 1-step sampling (THE MEANFLOW WAY!)
            samples = sample_meanflow_1step(
                model, num_samples=64, num_classes=10,
                device=device, cfg_scale=args.cfg_scale
            )
            
            # Save samples
            samples = (samples + 1) / 2  # Scale to [0, 1]
            torchvision.utils.save_image(
                samples, output_dir / f'samples_epoch_{epoch}_1step.png',
                nrow=8, normalize=False
            )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*70}")
    
    # Generate final samples - class-conditional
    print("\nGenerating class-conditional samples...")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for class_idx, class_name in enumerate(class_names):
        print(f"  Generating class {class_idx}: {class_name}")
        
        # Create class labels
        class_labels = torch.full((8,), class_idx, dtype=torch.long, device=device)
        
        # 1-step sampling
        samples = sample_meanflow_1step(
            model, num_samples=8, num_classes=10,
            device=device, cfg_scale=args.cfg_scale,
            class_labels=class_labels
        )
        
        # Save
        samples = (samples + 1) / 2
        torchvision.utils.save_image(
            samples, output_dir / f'final_class_{class_idx}_{class_name}.png',
            nrow=8, normalize=False
        )
    
    print("\n✓ All class-conditional samples generated!")
    print(f"✓ All outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train MeanFlow Matching on CIFAR-10')
    
    # Model architecture
    parser.add_argument('--base_channels', type=int, default=28,
                        help='Base number of channels (default: 28 for <1.5M params)')
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1, 2, 2, 2],
                        help='Channel multipliers at each resolution')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='Number of residual blocks per level')
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[16],
                        help='Resolutions at which to apply attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate for cosine schedule')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # MeanFlow specific
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.1,
                        help='Classifier-free guidance dropout probability')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--ratio_equal', type=float, default=0.25,
                        help='Ratio of samples where r=t (paper uses 0.25)')
    parser.add_argument('--loss_p', type=float, default=1.0,
                        help='Adaptive loss weighting power (0 for standard L2)')
    
    # Memory optimization
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs/meanflow',
                        help='Output directory for checkpoints and samples')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=10,
                        help='Generate samples every N epochs')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("MEANFLOW TRAINING CONFIGURATION")
    print("=" * 70)
    print("IMPLEMENTATION: True MeanFlow from paper arXiv:2505.13447")
    print("KEY FEATURES:")
    print("  - Model: u(z, r, t, y) with dual time variables")
    print("  - Training: JVP for MeanFlow Identity")
    print("  - Sampling: 1-step (no ODE integration!)")
    print("=" * 70)
    for key, value in vars(args).items():
        print(f"{key:30s}: {value}")
    print("=" * 70)
    
    train(args)


if __name__ == '__main__':
    main()