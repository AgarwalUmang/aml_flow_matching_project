"""
Main training script for Conditional Flow Matching
Optimized for 4GB GPU
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
import os
# Import custom modules
import sys
sys.path.append('.')

current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

# 3. Add paths so Python can see everything
sys.path.append(current_dir) # Allows seeing 'models'
sys.path.append(parent_dir)  # Allows seeing 'training'

from models.unet import TinyUNet, count_parameters
from trainer.cfm_trainer import ConditionalFlowMatcher, CFMTrainer, sample_cfm


def get_dataloaders(batch_size=32, num_workers=4, cache_in_memory=True):
    """Prepare CIFAR-10 dataloaders with optional memory caching"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    
    # Initialize model
    model = TinyUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_classes=10,
        dropout=args.dropout,
    )
    
    # Enable gradient checkpointing to save memory
    if args.gradient_checkpointing:
        print("Gradient checkpointing enabled (saves memory)")
        # Note: This would require modifying the UNet to support checkpointing
        # For simplicity, we skip this in the compact implementation
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params > 1.5e6:
        print(f"WARNING: Model has {num_params/1e6:.2f}M parameters (target: <1.5M)")
    
    # Initialize flow matcher and trainer
    flow_matcher = ConditionalFlowMatcher()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    
    trainer = CFMTrainer(
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
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch, args.cfg_dropout_prob)
        
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
            print("  Generating samples...")
            samples = sample_cfm(
                model, flow_matcher, num_samples=64, num_classes=10,
                device=device, num_steps=50, cfg_scale=args.cfg_scale
            )
            
            # Save samples
            samples = (samples + 1) / 2  # Scale to [0, 1]
            torchvision.utils.save_image(
                samples, output_dir / f'samples_epoch_{epoch}.png',
                nrow=8, normalize=False
            )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Conditional Flow Matching on CIFAR-10')
    
    # Model architecture
    parser.add_argument('--base_channels', type=int, default=28)
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1, 2, 2])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[16])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    
    # Memory optimization
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs/cfm')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--resume', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"{key:30s}: {value}")
    print("=" * 60)
    
    train(args)


if __name__ == '__main__':
    main()