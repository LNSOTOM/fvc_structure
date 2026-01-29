#!/usr/bin/env python3
"""
Visualize the 3D voxel data created from LiDAR and mask data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_and_visualize_voxels():
    """Load and visualize the voxel data"""
    
    # Load the tensor data
    print("Loading voxel tensors...")
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    
    occupancy = data['occupancy']
    point_counts = data['point_counts']
    dominant_labels = data['dominant_labels']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    print(f"Voxel grid shape: {occupancy.shape}")
    print(f"Voxel size: {voxel_size}")
    print(f"Origin: {origin}")
    print(f"Occupied voxels: {np.sum(occupancy):,}")
    
    # Load CSV for label information
    df = pd.read_csv("xyz_norm_with_labels.csv")
    unique_labels = df['Label'].dropna().unique()
    print(f"Unique labels: {unique_labels}")
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Overall occupancy (subsampled for performance)
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Find occupied voxels
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    
    # Subsample for visualization (take every nth point)
    subsample = slice(None, None, max(1, len(z_idx) // 5000))  # Max 5000 points
    z_sub = z_idx[subsample]
    y_sub = y_idx[subsample]
    x_sub = x_idx[subsample]
    
    ax1.scatter(x_sub, y_sub, z_sub, c=z_sub, cmap='viridis', alpha=0.6, s=1)
    ax1.set_xlabel('X (voxels)')
    ax1.set_ylabel('Y (voxels)')
    ax1.set_zlabel('Z (voxels)')
    ax1.set_title('3D Voxel Occupancy')
    
    # Plot 2: Point density histogram
    ax2 = fig.add_subplot(222)
    counts_nonzero = point_counts[point_counts > 0]
    ax2.hist(counts_nonzero, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Points per voxel')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Point Density Distribution')
    ax2.set_yscale('log')
    
    # Plot 3: Height distribution by label
    ax3 = fig.add_subplot(223)
    for label in unique_labels:
        if not np.isnan(label):
            label_data = df[df['Label'] == label]['Z_norm']
            ax3.hist(label_data, bins=30, alpha=0.6, label=f'Label {int(label)}')
    ax3.set_xlabel('Height (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Height Distribution by Label')
    ax3.legend()
    ax3.set_yscale('log')
    
    # Plot 4: Label distribution
    ax4 = fig.add_subplot(224)
    label_counts = df['Label'].value_counts(dropna=True)
    bars = ax4.bar(range(len(label_counts)), label_counts.values)
    ax4.set_xlabel('Label')
    ax4.set_ylabel('Point Count')
    ax4.set_title('Point Count by Label')
    ax4.set_xticks(range(len(label_counts)))
    ax4.set_xticklabels([f'Label {int(x)}' for x in label_counts.index])
    ax4.set_yscale('log')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, label_counts.values)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:,}', ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    plt.savefig('voxel_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a slice visualization
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show different Z slices
    nz = occupancy.shape[0]
    slice_indices = [nz//6, nz//3, nz//2, 2*nz//3, 5*nz//6, nz-1]
    
    for i, slice_idx in enumerate(slice_indices):
        ax = axes[i//3, i%3]
        
        # Show dominant labels for this slice
        slice_data = dominant_labels[slice_idx, :, :]
        occupied_slice = occupancy[slice_idx, :, :]
        
        # Mask out unoccupied voxels
        masked_data = np.where(occupied_slice > 0, slice_data, np.nan)
        
        im = ax.imshow(masked_data, cmap='tab10', vmin=0, vmax=4, alpha=0.8)
        ax.set_title(f'Z slice {slice_idx} (height: {slice_idx*voxel_size["z"]:.2f}m)')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        
        # Show only a subset for clarity
        ax.set_xlim(1000, 2000)
        ax.set_ylim(1000, 2000)
    
    plt.colorbar(im, ax=axes, label='Dominant Label')
    plt.tight_layout()
    plt.savefig('voxel_slices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization complete!")
    print("Saved: voxel_visualization.png, voxel_slices.png")

if __name__ == "__main__":
    load_and_visualize_voxels()