"""
Demo script for CFM Inpainting on CIFAR-10
Compatible with your training setup (checkpoint_best_cfm.pt)

Usage:
    python demo_inpainting.py --checkpoint checkpoint_best_cfm.pt
"""

import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

#Add paths so Python can see everything
sys.path.append(current_dir) 
sys.path.append(parent_dir)  

# Import your existing modules (assuming standard project structure)
from models.unet import TinyUNet
from trainer.cfm_trainer import ConditionalFlowMatcher

# Import inpainting functions
from cfm_inpainting import (
    sample_cfm_inpainting,
    sample_cfm_inpainting_rk4,
    InpaintingMaskGenerator,
    visualize_inpainting
)


def load_model(checkpoint_path, device='cuda'):
    """Load trained CFM model from your checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model args from checkpoint (saved during training)
    args = checkpoint.get('args', {})
    
    # Initialize model with same architecture as training
    model = TinyUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.get('base_channels', 32),
        channel_mults=args.get('channel_mults', [1, 2, 2]),
        num_res_blocks=args.get('num_res_blocks', 2),
        attention_resolutions=args.get('attention_resolutions', [16]),
        num_classes=10,
        dropout=args.get('dropout', 0.1),
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('train_loss', '?')
    print(f"✓ Model loaded (epoch {epoch}, loss {loss})")
    
    return model


def get_test_images(num_images=8):
    """Load test images from CIFAR-10"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Get random samples
    indices = torch.randperm(len(test_dataset))[:num_images]
    images = []
    labels = []
    
    for idx in indices:
        img, label = test_dataset[int(idx)]
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels


def demo_inpainting(args):
    """Run inpainting demo with all mask types"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    flow_matcher = ConditionalFlowMatcher()
    
    # Load test images
    print(f"Loading {args.num_images} test images...")
    images, labels = get_test_images(args.num_images)
    images = images.to(device)
    labels = labels.to(device)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    label_names = [class_names[l.item()] for l in labels]
    print(f" Images loaded, classes: {label_names}\n")
    
    # Create mask generator
    mask_gen = InpaintingMaskGenerator()
    
    # Test different mask types
    mask_configs = {
        'random_bbox': {
            'fn': lambda: mask_gen.random_bbox_mask(images.shape, min_size=8, max_size=20),
            'desc': 'Random bounding box masks'
        },
        'center': {
            'fn': lambda: mask_gen.center_mask(images.shape, mask_size=16),
            'desc': 'Center square mask (16x16)'
        },
        'irregular': {
            'fn': lambda: mask_gen.random_irregular_mask(images.shape, num_strokes=5),
            'desc': 'Irregular brush strokes'
        },
        'half_left': {
            'fn': lambda: mask_gen.half_image_mask(images.shape, direction='left'),
            'desc': 'Left half of image'
        },
    }
    
    for mask_name, mask_config in mask_configs.items():
        print(f"Processing: {mask_config['desc']}")
        
        # Generate mask
        mask = mask_config['fn']().to(device)
        
        # Create masked images
        masked_images = mask * images
        
        # Perform inpainting
        if args.use_rk4:
            print(f"  Using RK4 integration ({args.num_steps} steps)...")
            inpainted = sample_cfm_inpainting_rk4(
                model=model,
                flow_matcher=flow_matcher,
                image=images,
                mask=mask,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=labels if not args.unconditional else None,
                num_classes=10,
            )
        else:
            print(f"  Using Euler integration ({args.num_steps} steps, strategy: {args.resample_strategy})...")
            inpainted = sample_cfm_inpainting(
                model=model,
                flow_matcher=flow_matcher,
                image=images,
                mask=mask,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=labels if not args.unconditional else None,
                num_classes=10,
                resample_strategy=args.resample_strategy,
                jump_length=args.jump_length,
                num_resamples=args.num_resamples,
            )
        
        # Create visualization: [Original | Masked | Mask | Inpainted]
        vis = visualize_inpainting(images, masked_images, inpainted, mask)
        
        # Save visualization
        vis = (vis + 1) / 2  # Scale from [-1, 1] to [0, 1]
        save_path = output_dir / f'inpainting_{mask_name}.png'
        torchvision.utils.save_image(vis, save_path, nrow=1, padding=2)
        print(f"   Saved to {save_path}\n")
    
    print("=" * 60)
    print("Inpainting complete! ")
    print(f"Results saved to: {output_dir}")
    print(f"\nEach image shows: [Original | Masked | Mask | Inpainted]")
    print("=" * 60)


def demo_interactive(args):
    """Interactive mode: inpaint with specific mask type"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    flow_matcher = ConditionalFlowMatcher()
    
    # Load test images
    images, labels = get_test_images(args.num_images)
    images = images.to(device)
    labels = labels.to(device)
    
    mask_gen = InpaintingMaskGenerator()
    
    # User-specified mask type
    print(f"Mask type: {args.mask_type}")
    
    if args.mask_type == 'random_bbox':
        mask = mask_gen.random_bbox_mask(images.shape).to(device)
    elif args.mask_type == 'center':
        mask = mask_gen.center_mask(images.shape, mask_size=16).to(device)
    elif args.mask_type == 'irregular':
        mask = mask_gen.random_irregular_mask(images.shape).to(device)
    elif args.mask_type == 'half':
        mask = mask_gen.half_image_mask(images.shape, direction='left').to(device)
    else:
        raise ValueError(f"Unknown mask type: {args.mask_type}")
    
    masked_images = mask * images
    
    # Inpaint
    print(f"Inpainting with CFG scale={args.cfg_scale}, steps={args.num_steps}...")
    inpainted = sample_cfm_inpainting(
        model=model,
        flow_matcher=flow_matcher,
        image=images,
        mask=mask,
        device=device,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        class_labels=labels if not args.unconditional else None,
        num_classes=10,
        resample_strategy=args.resample_strategy,
    )
    
    # Visualize and save
    vis = visualize_inpainting(images, masked_images, inpainted, mask)
    vis = (vis + 1) / 2
    
    save_path = output_dir / f'inpainting_interactive_{args.mask_type}.png'
    torchvision.utils.save_image(vis, save_path, nrow=1, padding=2)
    print(f" Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='CFM Inpainting Demo')
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best_cfm.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_images', type=int, default=8,
                        help='Number of images to inpaint')
    parser.add_argument('--output_dir', type=str, default='./outputs/inpainting',
                        help='Output directory for results')
    
    # Inpainting parameters
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--unconditional', action='store_true',
                        help='Use unconditional generation (ignore class labels)')
    
    # Sampling strategy
    parser.add_argument('--use_rk4', action='store_true',
                        help='Use RK4 integration instead of Euler')
    parser.add_argument('--resample_strategy', type=str, default='replace',
                        choices=['replace', 'repaint'],
                        help='Resampling strategy for Euler method')
    parser.add_argument('--jump_length', type=int, default=5,
                        help='Jump length for repaint strategy')
    parser.add_argument('--num_resamples', type=int, default=3,
                        help='Number of resamples for repaint strategy')
    
    # Mode
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'interactive'],
                        help='Demo mode (all masks) or interactive mode (specific mask)')
    parser.add_argument('--mask_type', type=str, default='random_bbox',
                        choices=['random_bbox', 'center', 'irregular', 'half'],
                        help='Mask type for interactive mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CFM Image Inpainting Demo")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of images: {args.num_images}")
    print(f"Sampling steps: {args.num_steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Integration: {'RK4' if args.use_rk4 else 'Euler'}")
    if not args.use_rk4:
        print(f"Strategy: {args.resample_strategy}")
    print("=" * 60 + "\n")
    
    if args.mode == 'demo':
        demo_inpainting(args)
    else:
        demo_interactive(args)


if __name__ == '__main__':
    main()