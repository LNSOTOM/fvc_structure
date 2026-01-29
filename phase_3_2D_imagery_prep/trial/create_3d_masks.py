#!/usr/bin/env python3
"""
Create 3D training masks for U-Net following Z slices with labels 0-3,
Z_norm values, and voxel occupancy information.
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

def create_3d_unet_masks():
    """Create 3D training masks with labels, Z_norm, and occupancy"""
    
    # Load the CSV data
    print("Loading point cloud data...")
    csv_path = "xyz_norm_with_labels.csv"
    df = pd.read_csv(csv_path)
    
    # Load voxel tensors for reference
    print("Loading voxel tensors...")
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    # Load mask to get transform
    mask_path = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/dowsample_5cm_bilinear/mask_fvc_3072.26.tif"
    with rasterio.open(mask_path) as src:
        transform = src.transform
        
    print(f"Voxel grid shape: {occupancy.shape}")
    print(f"Voxel size: {voxel_size}")
    print(f"Z range from data: {df['Z_norm'].min():.3f} to {df['Z_norm'].max():.3f}m")
    
    # Parameters
    vs_z = voxel_size['z']   # 0.05m
    z_min = origin['z']
    
    nz, ny, nx = occupancy.shape
    
    # Create multiple 3D masks
    print("Creating 3D masks...")
    
    # 1. Label mask (0-3 vegetation classes)
    label_mask = dominant_labels.copy().astype(np.uint8)
    label_mask[occupancy == 0] = 255  # NoData for empty voxels
    
    # 2. Z_norm mask (height values scaled to 0-255)
    z_coords = np.zeros_like(occupancy, dtype=np.float32)
    for z in range(nz):
        z_coords[z, :, :] = z_min + z * vs_z
    z_norm_mask = z_coords.copy()
    z_norm_mask[occupancy == 0] = -9999  # NoData for empty voxels
    
    # 3. Occupancy mask (binary: 0=empty, 1=occupied)
    occupancy_mask = occupancy.copy()
    
    # 4. Multi-class mask combining labels and occupancy
    # 0: empty, 1-4: vegetation classes (shifted by 1)
    multiclass_mask = np.zeros_like(occupancy, dtype=np.uint8)
    multiclass_mask[occupancy > 0] = dominant_labels[occupancy > 0] + 1
    
    # Create Z-slice specific masks for different height ranges
    print("Creating height-stratified masks...")
    
    height_ranges = [
        (0.0, 0.5, "ground"),      # 0-50cm
        (0.5, 2.0, "understory"),  # 50cm-2m
        (2.0, 5.0, "canopy"),      # 2m-5m
        (5.0, 20.0, "overstory")   # >5m
    ]
    
    stratified_masks = {}
    for z_min_h, z_max_h, layer_name in height_ranges:
        mask = np.zeros((ny, nx), dtype=np.uint8)
        
        # Find Z indices for this height range
        z_start = max(0, int((z_min_h - z_min) / vs_z))
        z_end = min(nz, int((z_max_h - z_min) / vs_z))
        
        if z_start < z_end:
            # Aggregate labels in this height range
            layer_data = dominant_labels[z_start:z_end, :, :]
            layer_occ = occupancy[z_start:z_end, :, :]
            
            # For each pixel, find the dominant label in this height range
            for y in range(ny):
                for x in range(nx):
                    column_data = layer_data[:, y, x]
                    column_occ = layer_occ[:, y, x]
                    
                    if np.any(column_occ > 0):
                        # Get labels from occupied voxels only
                        valid_labels = column_data[column_occ > 0]
                        if len(valid_labels) > 0:
                            # Most common label
                            unique, counts = np.unique(valid_labels, return_counts=True)
                            mask[y, x] = unique[np.argmax(counts)]
                        else:
                            mask[y, x] = 255  # NoData
                    else:
                        mask[y, x] = 255  # NoData
        
        stratified_masks[layer_name] = mask
        print(f"  {layer_name}: {z_min_h}-{z_max_h}m, Z indices {z_start}-{z_end}")
    
    # Save all masks
    print("Saving 3D masks...")
    
    # Save 3D tensors
    np.savez_compressed(
        "unet_training_masks.npz",
        label_mask=label_mask,
        z_norm_mask=z_norm_mask,
        occupancy_mask=occupancy_mask,
        multiclass_mask=multiclass_mask,
        stratified_masks=stratified_masks,
        voxel_size=voxel_size,
        origin=origin,
        shape=occupancy.shape,
        height_ranges=height_ranges
    )
    
    # Save 2D stratified masks as GeoTIFFs
    print("Saving stratified masks as GeoTIFFs...")
    
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 255,
        "width": nx,
        "height": ny,
        "count": 1,
        "transform": transform,
        "compress": "lzw",
    }
    
    for layer_name, mask in stratified_masks.items():
        output_file = f"mask_3d_{layer_name}.tif"
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(mask, 1)
        print(f"  Saved: {output_file}")
    
    # Create visualization
    print("Creating visualizations...")
    
    # Plot 1: Z-slice progression
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Show sample Z slices
    z_indices = [5, 15, 25, 35, 45, 55, 65, 75]  # Different heights
    
    for i, z_idx in enumerate(z_indices):
        if z_idx < nz:
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            # Show labels with occupancy overlay
            slice_data = label_mask[z_idx, :, :]
            slice_occ = occupancy_mask[z_idx, :, :]
            
            # Crop to data region
            y_min, y_max = 1000, 2000
            x_min, x_max = 1000, 2000
            
            cropped_data = slice_data[y_min:y_max, x_min:x_max]
            cropped_occ = slice_occ[y_min:y_max, x_min:x_max]
            
            # Mask empty areas
            display_data = np.where(cropped_occ > 0, cropped_data, 255)
            
            ax.imshow(display_data, cmap='tab10', vmin=0, vmax=4)
            ax.set_title(f'Z={z_idx*vs_z:.2f}m')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('z_slice_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Stratified masks
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, (layer_name, mask) in enumerate(stratified_masks.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Crop to data region
        cropped_mask = mask[1000:2000, 1000:2000]
        
        ax.imshow(cropped_mask, cmap='tab10', vmin=0, vmax=4)
        ax.set_title(f'{layer_name.title()} Layer')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('stratified_masks.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistics
    print("\nMask Statistics:")
    print(f"3D Label mask shape: {label_mask.shape}")
    print(f"Occupied voxels: {np.sum(occupancy_mask):,}")
    print(f"Total voxels: {np.prod(occupancy_mask.shape):,}")
    print(f"Occupancy rate: {100*np.sum(occupancy_mask)/np.prod(occupancy_mask.shape):.2f}%")
    
    print("\nLabel distribution in 3D:")
    unique_labels, counts = np.unique(label_mask[occupancy_mask > 0], return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count:,} voxels ({100*count/np.sum(counts):.1f}%)")
    
    print("\nStratified mask statistics:")
    for layer_name, mask in stratified_masks.items():
        occupied_pixels = np.sum(mask != 255)
        total_pixels = mask.size
        print(f"  {layer_name}: {occupied_pixels:,}/{total_pixels:,} pixels ({100*occupied_pixels/total_pixels:.2f}%)")
    
    print("\nFiles created:")
    print("- unet_training_masks.npz (3D tensors)")
    print("- mask_3d_*.tif (stratified 2D masks)")
    print("- z_slice_progression.png (visualization)")
    print("- stratified_masks.png (visualization)")
    
    return {
        'label_mask': label_mask,
        'z_norm_mask': z_norm_mask,
        'occupancy_mask': occupancy_mask,
        'multiclass_mask': multiclass_mask,
        'stratified_masks': stratified_masks
    }

if __name__ == "__main__":
    masks = create_3d_unet_masks()