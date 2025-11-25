"""
Comprehensive Inpainting Evaluation on CIFAR-10
Evaluates inpainting quality on 500 images per class (5000 total)
Reports NMSE, PSNR, SSIM metrics per class and overall
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import numpy as np
import json
from tqdm import tqdm
import time
from collections import defaultdict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

#Add paths so Python can see everything
sys.path.append(current_dir) 
sys.path.append(parent_dir)  

from models.unet import TinyUNet
from trainer.cfm_trainer import ConditionalFlowMatcher
from cfm_inpainting import (
    sample_cfm_inpainting,
    sample_cfm_inpainting_rk4,
    InpaintingMaskGenerator,
)


def calculate_nmse(original, inpainted, mask):
    """
    Calculate Normalized Mean Squared Error on masked regions
    
    Args:
        original: (B, C, H, W) in [-1, 1]
        inpainted: (B, C, H, W) in [-1, 1]
        mask: (B, 1, H, W) where 0 = inpainted region
    
    Returns:
        nmse: scalar
    """
    # Focus only on inpainted regions (where mask == 0)
    inpainted_region = 1 - mask
    
    # MSE on inpainted region
    mse = ((original - inpainted) ** 2 * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    
    # Normalize by variance of original
    variance = (original ** 2 * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    
    nmse = mse / (variance + 1e-8)
    
    return nmse.item()


def calculate_psnr(original, inpainted, mask, data_range=2.0):
    """
    Calculate Peak Signal-to-Noise Ratio on masked regions
    
    Args:
        original: (B, C, H, W) in [-1, 1]
        inpainted: (B, C, H, W) in [-1, 1]
        mask: (B, 1, H, W) where 0 = inpainted region
        data_range: range of data (2.0 for [-1, 1])
    
    Returns:
        psnr: scalar in dB
    """
    # Focus only on inpainted regions
    inpainted_region = 1 - mask
    
    # MSE on inpainted region
    mse = ((original - inpainted) ** 2 * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    
    # PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse + 1e-8)
    
    return psnr.item()


def calculate_ssim(original, inpainted, mask):
    """
    Calculate Structural Similarity Index on masked regions
    Simplified SSIM for efficiency
    
    Args:
        original: (B, C, H, W) in [-1, 1]
        inpainted: (B, C, H, W) in [-1, 1]
        mask: (B, 1, H, W) where 0 = inpainted region
    
    Returns:
        ssim: scalar between -1 and 1
    """
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2
    
    # Only compute on masked regions
    inpainted_region = 1 - mask
    
    # Mean
    mu1 = (original * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    mu2 = (inpainted * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    
    # Variance
    sigma1_sq = ((original - mu1) ** 2 * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    sigma2_sq = ((inpainted - mu2) ** 2 * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    sigma12 = ((original - mu1) * (inpainted - mu2) * inpainted_region).sum() / (inpainted_region.sum() + 1e-8)
    
    # SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()


def load_model(checkpoint_path, device='cuda'):
    """Load trained CFM model"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
    return model


def get_stratified_dataset(num_per_class=500):
    """
    Get stratified dataset with exactly num_per_class samples per class
    
    Returns:
        images: list of tensors
        labels: list of ints
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load CIFAR-10 test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Organize by class
    class_samples = defaultdict(list)
    for idx, (img, label) in enumerate(test_dataset):
        if len(class_samples[label]) < num_per_class:
            class_samples[label].append((img, label))
        
        # Check if we have enough samples for all classes
        if all(len(samples) >= num_per_class for samples in class_samples.values()):
            break
    
    # Flatten
    all_images = []
    all_labels = []
    for label in range(10):
        for img, lbl in class_samples[label]:
            all_images.append(img)
            all_labels.append(lbl)
    
    print(f"✓ Loaded {len(all_images)} images ({num_per_class} per class)")
    return all_images, all_labels


def evaluate_single_mask_type(model, flow_matcher, all_images, all_labels, 
                              mask_gen, mask_type, args, device, class_names):
    """Evaluate on a single mask type"""
    total_samples = len(all_images)
    
    # Results storage
    results = {
        'per_class': defaultdict(lambda: {
            'nmse': [],
            'psnr': [],
            'ssim': [],
            'count': 0,
        }),
        'overall': {
            'nmse': [],
            'psnr': [],
            'ssim': [],
        },
    }
    
    # Process in batches
    start_time = time.time()
    
    for batch_start in tqdm(range(0, total_samples, args.batch_size), 
                            desc=f"Evaluating {mask_type}"):
        batch_end = min(batch_start + args.batch_size, total_samples)
        
        # Get batch
        batch_images = torch.stack([all_images[i] for i in range(batch_start, batch_end)])
        batch_labels = torch.tensor([all_labels[i] for i in range(batch_start, batch_end)])
        
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # Generate masks
        if mask_type == 'random_bbox':
            masks = mask_gen.random_bbox_mask(batch_images.shape, min_size=8, max_size=20).to(device)
        elif mask_type == 'center':
            masks = mask_gen.center_mask(batch_images.shape, mask_size=16).to(device)
        elif mask_type == 'irregular':
            masks = mask_gen.random_irregular_mask(batch_images.shape, num_strokes=5).to(device)
        elif mask_type == 'half':
            masks = mask_gen.half_image_mask(batch_images.shape, direction='left').to(device)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Perform inpainting
        if args.use_rk4:
            inpainted = sample_cfm_inpainting_rk4(
                model=model,
                flow_matcher=flow_matcher,
                image=batch_images,
                mask=masks,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=batch_labels,
                num_classes=10,
            )
        else:
            inpainted = sample_cfm_inpainting(
                model=model,
                flow_matcher=flow_matcher,
                image=batch_images,
                mask=masks,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=batch_labels,
                num_classes=10,
                resample_strategy=args.resample_strategy,
            )
        
        # Calculate metrics for each image in batch
        for i in range(len(batch_images)):
            original = batch_images[i:i+1]
            inpainted_img = inpainted[i:i+1]
            mask = masks[i:i+1]
            label = batch_labels[i].item()
            
            # Compute metrics
            nmse = calculate_nmse(original, inpainted_img, mask)
            psnr = calculate_psnr(original, inpainted_img, mask)
            ssim = calculate_ssim(original, inpainted_img, mask)
            
            # Store results
            results['per_class'][label]['nmse'].append(nmse)
            results['per_class'][label]['psnr'].append(psnr)
            results['per_class'][label]['ssim'].append(ssim)
            results['per_class'][label]['count'] += 1
            
            results['overall']['nmse'].append(nmse)
            results['overall']['psnr'].append(psnr)
            results['overall']['ssim'].append(ssim)
    
    elapsed_time = time.time() - start_time
    
    # Compute statistics
    summary_stats = {
        'per_class': {},
        'overall': {},
    }
    
    for label in range(10):
        class_name = class_names[label]
        count = results['per_class'][label]['count']
        
        if count > 0:
            nmse_mean = np.mean(results['per_class'][label]['nmse'])
            nmse_std = np.std(results['per_class'][label]['nmse'])
            psnr_mean = np.mean(results['per_class'][label]['psnr'])
            psnr_std = np.std(results['per_class'][label]['psnr'])
            ssim_mean = np.mean(results['per_class'][label]['ssim'])
            ssim_std = np.std(results['per_class'][label]['ssim'])
            
            summary_stats['per_class'][class_name] = {
                'count': count,
                'nmse_mean': float(nmse_mean),
                'nmse_std': float(nmse_std),
                'psnr_mean': float(psnr_mean),
                'psnr_std': float(psnr_std),
                'ssim_mean': float(ssim_mean),
                'ssim_std': float(ssim_std),
            }
    
    # Overall statistics
    nmse_overall = np.mean(results['overall']['nmse'])
    nmse_overall_std = np.std(results['overall']['nmse'])
    psnr_overall = np.mean(results['overall']['psnr'])
    psnr_overall_std = np.std(results['overall']['psnr'])
    ssim_overall = np.mean(results['overall']['ssim'])
    ssim_overall_std = np.std(results['overall']['ssim'])
    
    summary_stats['overall'] = {
        'total_samples': total_samples,
        'nmse_mean': float(nmse_overall),
        'nmse_std': float(nmse_overall_std),
        'psnr_mean': float(psnr_overall),
        'psnr_std': float(psnr_overall_std),
        'ssim_mean': float(ssim_overall),
        'ssim_std': float(ssim_overall_std),
        'processing_time_seconds': float(elapsed_time),
        'time_per_image_seconds': float(elapsed_time / total_samples),
    }
    
    return summary_stats, elapsed_time


def print_results_table(mask_type, summary_stats, class_names):
    """Print results table for a single mask type"""
    print(f"\n{'='*60}")
    print(f"RESULTS FOR: {mask_type.upper()}")
    print('='*60)
    print(f"{'Class':<12} {'Count':>6} {'NMSE':>10} {'PSNR (dB)':>12} {'SSIM':>10}")
    print('-'*60)
    
    for class_name in class_names:
        if class_name in summary_stats['per_class']:
            stats = summary_stats['per_class'][class_name]
            print(f"{class_name:<12} {stats['count']:>6} {stats['nmse_mean']:>10.4f} "
                  f"{stats['psnr_mean']:>12.2f} {stats['ssim_mean']:>10.4f}")
    
    print('-'*60)
    overall = summary_stats['overall']
    print(f"{'OVERALL':<12} {overall['total_samples']:>6} {overall['nmse_mean']:>10.4f} "
          f"{overall['psnr_mean']:>12.2f} {overall['ssim_mean']:>10.4f}")
    print('='*60)


def print_comparison_table(all_results, class_names):
    """Print comparison table across all mask types"""
    mask_types = list(all_results.keys())
    
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS ALL MASK TYPES")
    print('='*80)
    
    # Per-class comparison
    print(f"\n{'Class':<12}", end="")
    for mask_type in mask_types:
        print(f" | {mask_type[:6]:>18}", end="")
    print()
    print(f"{'':>12}", end="")
    for _ in mask_types:
        print(f" | {'NMSE':>6} {'PSNR':>6} {'SSIM':>5}", end="")
    print()
    print('-'*80)
    
    for class_name in class_names:
        print(f"{class_name:<12}", end="")
        for mask_type in mask_types:
            if class_name in all_results[mask_type]['per_class']:
                stats = all_results[mask_type]['per_class'][class_name]
                print(f" | {stats['nmse_mean']:>6.4f} {stats['psnr_mean']:>6.1f} {stats['ssim_mean']:>5.3f}", end="")
            else:
                print(f" | {'N/A':>6} {'N/A':>6} {'N/A':>5}", end="")
        print()
    
    # Overall comparison
    print('-'*80)
    print(f"{'OVERALL':<12}", end="")
    for mask_type in mask_types:
        overall = all_results[mask_type]['overall']
        print(f" | {overall['nmse_mean']:>6.4f} {overall['psnr_mean']:>6.1f} {overall['ssim_mean']:>5.3f}", end="")
    print()
    print('='*80)
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print('-'*80)
    print(f"{'Mask Type':<15} {'NMSE':>10} {'PSNR (dB)':>12} {'SSIM':>10} {'Time (s)':>10}")
    print('-'*80)
    for mask_type in mask_types:
        overall = all_results[mask_type]['overall']
        print(f"{mask_type:<15} {overall['nmse_mean']:>10.4f} {overall['psnr_mean']:>12.2f} "
              f"{overall['ssim_mean']:>10.4f} {overall['processing_time_seconds']:>10.1f}")
    print('='*80)


def evaluate_inpainting(args):
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    flow_matcher = ConditionalFlowMatcher()
    
    # Load dataset
    print(f"\nLoading dataset ({args.num_per_class} samples per class)...")
    all_images, all_labels = get_stratified_dataset(args.num_per_class)
    total_samples = len(all_images)
    
    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Mask generator
    mask_gen = InpaintingMaskGenerator()
    
    # Determine which mask types to evaluate
    if args.mask_type is None:
        # Evaluate all mask types
        mask_types = ['center', 'random_bbox', 'irregular', 'half']
        print(f"\n No mask type specified - evaluating ALL mask types!")
        print(f"Total evaluations: {len(mask_types)} mask types × {total_samples} images = {len(mask_types) * total_samples} total")
    else:
        # Single mask type
        mask_types = [args.mask_type]
        print(f"\n Evaluating single mask type: {args.mask_type}")
    
    print(f"\nConfiguration:")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Integration: {'RK4' if args.use_rk4 else 'Euler'}")
    print(f"  Batch size: {args.batch_size}")
    print("="*60)
    
    # Evaluate each mask type
    all_results = {}
    total_start_time = time.time()
    
    for mask_type in mask_types:
        print(f"\n{'='*60}")
        print(f"Processing: {mask_type.upper()}")
        print('='*60)
        
        summary_stats, elapsed_time = evaluate_single_mask_type(
            model, flow_matcher, all_images, all_labels,
            mask_gen, mask_type, args, device, class_names
        )
        
        all_results[mask_type] = summary_stats
        
        # Print results for this mask type
        print_results_table(mask_type, summary_stats, class_names)
        print(f"\nTime for {mask_type}: {elapsed_time:.1f}s ({elapsed_time/total_samples:.2f}s per image)")
    
    total_elapsed_time = time.time() - total_start_time
    
    # Print comparison if multiple mask types
    if len(mask_types) > 1:
        print_comparison_table(all_results, class_names)
    
    # Save results
    if len(mask_types) == 1:
        # Single mask type - save as before
        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results[mask_types[0]], f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
    else:
        # Multiple mask types - save combined results
        results_file = output_dir / 'evaluation_results_all_masks.json'
        combined_results = {
            'mask_types': mask_types,
            'results': all_results,
            'config': vars(args),
            'total_time_seconds': float(total_elapsed_time),
        }
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\n Combined results saved to: {results_file}")
        
        # Also save individual files for each mask
        for mask_type in mask_types:
            mask_file = output_dir / f'evaluation_results_{mask_type}.json'
            with open(mask_file, 'w') as f:
                json.dump(all_results[mask_type], f, indent=2)
            print(f" {mask_type} results saved to: {mask_file}")
    
    # Create summary report
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CFM INPAINTING EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  Mask types: {', '.join(mask_types)}\n")
        f.write(f"  CFG scale: {args.cfg_scale}\n")
        f.write(f"  Num steps: {args.num_steps}\n")
        f.write(f"  Integration: {'RK4' if args.use_rk4 else 'Euler'}\n")
        f.write(f"  Strategy: {args.resample_strategy}\n\n")
        
        f.write("Dataset:\n")
        f.write(f"  Total images per mask: {total_samples}\n")
        f.write(f"  Images per class: {args.num_per_class}\n")
        f.write(f"  Total processing time: {total_elapsed_time:.1f}s\n\n")
        
        for mask_type in mask_types:
            f.write(f"\n{'='*60}\n")
            f.write(f"Results for: {mask_type.upper()}\n")
            f.write('='*60 + "\n\n")
            
            overall = all_results[mask_type]['overall']
            f.write(f"Overall Results:\n")
            f.write(f"  NMSE:  {overall['nmse_mean']:.4f} ± {overall['nmse_std']:.4f}\n")
            f.write(f"  PSNR:  {overall['psnr_mean']:.2f} ± {overall['psnr_std']:.2f} dB\n")
            f.write(f"  SSIM:  {overall['ssim_mean']:.4f} ± {overall['ssim_std']:.4f}\n")
            f.write(f"  Time:  {overall['processing_time_seconds']:.1f}s\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("Per-Class Results:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Class':<12} {'NMSE':>10} {'PSNR':>12} {'SSIM':>10}\n")
            f.write("-"*60 + "\n")
            
            for class_name in class_names:
                if class_name in all_results[mask_type]['per_class']:
                    stats = all_results[mask_type]['per_class'][class_name]
                    f.write(f"{class_name:<12} {stats['nmse_mean']:>10.4f} "
                           f"{stats['psnr_mean']:>12.2f} {stats['ssim_mean']:>10.4f}\n")
        
        if len(mask_types) > 1:
            f.write("\n\n" + "="*80 + "\n")
            f.write("COMPARISON ACROSS ALL MASK TYPES\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Mask Type':<15} {'NMSE':>10} {'PSNR (dB)':>12} {'SSIM':>10}\n")
            f.write("-"*80 + "\n")
            for mask_type in mask_types:
                overall = all_results[mask_type]['overall']
                f.write(f"{mask_type:<15} {overall['nmse_mean']:>10.4f} "
                       f"{overall['psnr_mean']:>12.2f} {overall['ssim_mean']:>10.4f}\n")
    
    print(f" Report saved to: {report_file}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Total time: {total_elapsed_time:.1f}s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CFM Inpainting on CIFAR-10')
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best_cfm.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_per_class', type=int, default=500,
                        help='Number of samples per class (default: 500 = 5000 total)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation',
                        help='Output directory for results')
    
    # Inpainting parameters
    parser.add_argument('--mask_type', type=str, default=None,
                        choices=['random_bbox', 'center', 'irregular', 'half', None],
                        help='Type of mask to use (if not specified, evaluates all mask types)')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='CFG scale')
    parser.add_argument('--use_rk4', action='store_true',
                        help='Use RK4 integration')
    parser.add_argument('--resample_strategy', type=str, default='replace',
                        choices=['replace', 'repaint'],
                        help='Resampling strategy')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CFM INPAINTING EVALUATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {args.num_per_class} per class = {args.num_per_class * 10} total")
    print(f"  Mask type: {args.mask_type if args.mask_type else 'ALL (center, random_bbox, irregular, half)'}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Integration: {'RK4' if args.use_rk4 else 'Euler'}")
    print("="*60 + "\n")
    
    evaluate_inpainting(args)


if __name__ == '__main__':
    main()