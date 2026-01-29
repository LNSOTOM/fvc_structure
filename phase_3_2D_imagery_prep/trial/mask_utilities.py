#!/usr/bin/env python3
"""
Utility functions to load and work with the 3D U-Net training masks
"""

import numpy as np
import matplotlib.pyplot as plt

def load_training_masks():
    """Load all training masks from the .npz file"""
    data = np.load("unet_training_masks.npz", allow_pickle=True)
    return {
        'label_mask': data['label_mask'],           # 3D labels (0-3, 255=nodata)
        'z_norm_mask': data['z_norm_mask'],         # 3D height values
        'occupancy_mask': data['occupancy_mask'],   # 3D binary occupancy
        'multiclass_mask': data['multiclass_mask'], # 3D labels shifted (0=empty, 1-4=classes)
        'stratified_masks': data['stratified_masks'].item(), # 2D stratified by height
        'voxel_size': data['voxel_size'].item(),
        'origin': data['origin'].item(),
        'shape': data['shape'],
        'height_ranges': data['height_ranges']
    }

def extract_z_slice(masks, z_height_m):
    """Extract a Z slice at specific height for training"""
    voxel_size = masks['voxel_size']
    origin = masks['origin']
    vs_z = voxel_size['z']
    z_min = origin['z']
    
    # Convert height to Z index
    z_idx = int((z_height_m - z_min) / vs_z)
    z_idx = max(0, min(z_idx, masks['label_mask'].shape[0] - 1))
    
    slice_data = {
        'z_index': z_idx,
        'z_height': z_min + z_idx * vs_z,
        'labels': masks['label_mask'][z_idx, :, :],
        'occupancy': masks['occupancy_mask'][z_idx, :, :],
        'z_norm': masks['z_norm_mask'][z_idx, :, :],
        'multiclass': masks['multiclass_mask'][z_idx, :, :]
    }
    
    return slice_data

def get_training_patches(masks, patch_size=128, stride=64, min_occupancy=0.1):
    """Extract training patches from 3D masks"""
    label_mask = masks['label_mask']
    occupancy_mask = masks['occupancy_mask']
    
    nz, ny, nx = label_mask.shape
    patches = []
    
    for z in range(0, nz, stride):
        for y in range(0, ny - patch_size + 1, stride):
            for x in range(0, nx - patch_size + 1, stride):
                # Extract patch
                z_end = min(z + patch_size, nz)
                y_end = y + patch_size
                x_end = x + patch_size
                
                patch_labels = label_mask[z:z_end, y:y_end, x:x_end]
                patch_occupancy = occupancy_mask[z:z_end, y:y_end, x:x_end]
                
                # Check if patch has sufficient data
                occupancy_rate = np.sum(patch_occupancy) / patch_occupancy.size
                
                if occupancy_rate >= min_occupancy:
                    patches.append({
                        'coords': (z, y, x),
                        'size': (z_end - z, y_end - y, x_end - x),
                        'labels': patch_labels,
                        'occupancy': patch_occupancy,
                        'occupancy_rate': occupancy_rate
                    })
    
    return patches

def visualize_z_progression(masks, z_heights=None, figsize=(15, 10)):
    """Visualize label progression through Z slices"""
    if z_heights is None:
        # Default heights every 50cm
        z_heights = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    n_heights = len(z_heights)
    cols = 4
    rows = (n_heights + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, height in enumerate(z_heights):
        if i >= rows * cols:
            break
            
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        slice_data = extract_z_slice(masks, height)
        
        # Crop to data region
        y_start, y_end = 1000, 2000
        x_start, x_end = 1000, 2000
        
        cropped_labels = slice_data['labels'][y_start:y_end, x_start:x_end]
        cropped_occ = slice_data['occupancy'][y_start:y_end, x_start:x_end]
        
        # Mask empty areas
        display_data = np.where(cropped_occ > 0, cropped_labels, 255)
        
        ax.imshow(display_data, cmap='tab10', vmin=0, vmax=4)
        ax.set_title(f'Height: {height:.1f}m\n(Z-idx: {slice_data["z_index"]})')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for i in range(len(z_heights), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('z_progression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_mask_summary():
    """Print summary of available masks"""
    masks = load_training_masks()
    
    print("=== 3D U-Net Training Masks Summary ===\n")
    
    print(f"Voxel Grid Shape: {masks['shape']}")
    print(f"Voxel Size: {masks['voxel_size']} meters")
    print(f"Origin: {masks['origin']}")
    
    occupancy = np.sum(masks['occupancy_mask'])
    total = np.prod(masks['shape'])
    print(f"Occupancy: {occupancy:,} / {total:,} voxels ({100*occupancy/total:.2f}%)")
    
    print("\n--- Available Mask Types ---")
    print("1. label_mask: Labels 0-3 for vegetation classes (255=nodata)")
    print("2. z_norm_mask: Height values in meters (-9999=nodata)")
    print("3. occupancy_mask: Binary occupancy (0=empty, 1=occupied)")
    print("4. multiclass_mask: Labels shifted (0=empty, 1-4=vegetation classes)")
    
    print("\n--- Height Stratified Masks (2D) ---")
    for layer_name, mask in masks['stratified_masks'].items():
        occupied = np.sum(mask != 255)
        total_2d = mask.size
        print(f"- {layer_name}: {occupied:,} / {total_2d:,} pixels ({100*occupied/total_2d:.2f}%)")
    
    print("\n--- Label Distribution in 3D ---")
    occupied_voxels = masks['occupancy_mask'] > 0
    valid_labels = masks['label_mask'][occupied_voxels]
    unique, counts = np.unique(valid_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count:,} voxels ({100*count/len(valid_labels):.1f}%)")

def create_training_sample():
    """Create a sample training batch"""
    masks = load_training_masks()
    
    # Get some training patches
    patches = get_training_patches(masks, patch_size=64, stride=32, min_occupancy=0.05)
    
    print(f"Found {len(patches)} training patches")
    
    if len(patches) > 0:
        # Show first few patches
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for i in range(min(6, len(patches))):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            patch = patches[i]
            # Show middle Z slice of the patch
            mid_z = patch['labels'].shape[0] // 2
            
            slice_labels = patch['labels'][mid_z, :, :]
            slice_occ = patch['occupancy'][mid_z, :, :]
            
            display_data = np.where(slice_occ > 0, slice_labels, 255)
            
            ax.imshow(display_data, cmap='tab10', vmin=0, vmax=4)
            ax.set_title(f'Patch {i+1}\nOcc: {patch["occupancy_rate"]:.1%}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('training_patches_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return patches

if __name__ == "__main__":
    print_mask_summary()
    print("\n" + "="*50)
    
    # Load masks
    masks = load_training_masks()
    
    # Visualize Z progression
    print("Creating Z progression visualization...")
    visualize_z_progression(masks)
    
    # Create training sample
    print("Creating training patches sample...")
    patches = create_training_sample()
    
    print(f"\nCreated {len(patches)} training patches")
    print("Files saved:")
    print("- z_progression_analysis.png")
    print("- training_patches_sample.png")