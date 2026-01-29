#!/usr/bin/env python3
"""
Convert voxel occupancy data to 3D volumes and point clouds for visualization
in CloudCompare, MeshLab, or other 3D tools.
"""

import numpy as np
import pandas as pd
import open3d as o3d

def convert_voxels_to_pointcloud():
    """Convert occupied voxels to a colored point cloud"""
    
    print("Loading voxel data...")
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    point_counts = data['point_counts']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    print(f"Voxel grid shape: {occupancy.shape}")
    print(f"Voxel size: {voxel_size}")
    
    # Get occupied voxel coordinates
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    
    print(f"Found {len(z_idx)} occupied voxels")
    
    # Convert voxel indices to world coordinates
    vs_x, vs_y, vs_z = voxel_size['x'], voxel_size['y'], voxel_size['z']
    origin_x, origin_y, origin_z = origin['x'], origin['y'], origin['z']
    
    # Calculate world coordinates (center of each voxel)
    world_x = origin_x + x_idx * vs_x + vs_x/2
    world_y = origin_y + y_idx * vs_y + vs_y/2  
    world_z = origin_z + z_idx * vs_z + vs_z/2
    
    # Get labels and point counts for occupied voxels
    labels = dominant_labels[z_idx, y_idx, x_idx]
    counts = point_counts[z_idx, y_idx, x_idx]
    
    # Create point cloud
    points = np.column_stack([world_x, world_y, world_z])
    
    # Create colors based on labels
    colors = np.zeros((len(points), 3))
    color_map = {
        0: [0.8, 0.4, 0.2],   # Brown for label 0 (bare ground)
        1: [0.9, 0.9, 0.1],   # Yellow for label 1 (litter/non-photosynthetic)
        2: [0.2, 0.8, 0.2],   # Green for label 2 (photosynthetic vegetation)
        3: [0.1, 0.1, 0.7],   # Blue for label 3 (shadow)
    }
    
    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = color_map[label]
        else:
            colors[i] = [0.5, 0.5, 0.5]  # Gray for unknown
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as PLY (readable by CloudCompare, MeshLab, etc.)
    ply_file = "voxel_occupancy_pointcloud.ply"
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"✓ Saved point cloud: {ply_file}")
    
    # Save as CSV with all attributes
    df = pd.DataFrame({
        'X': world_x,
        'Y': world_y, 
        'Z': world_z,
        'Label': labels,
        'PointCount': counts,
        'R': (colors[:, 0] * 255).astype(int),
        'G': (colors[:, 1] * 255).astype(int),
        'B': (colors[:, 2] * 255).astype(int)
    })
    
    csv_file = "voxel_occupancy_pointcloud.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    
    return pcd, df

def create_voxel_mesh():
    """Create a 3D mesh from voxels for visualization"""
    
    print("Creating voxel mesh...")
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    # Get occupied voxels
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    labels = dominant_labels[z_idx, y_idx, x_idx]
    
    vs = voxel_size['x']  # Assume cubic voxels
    origin_x, origin_y, origin_z = origin['x'], origin['y'], origin['z']
    
    # Create individual voxel meshes and combine
    mesh_list = []
    color_map = {
        0: [0.8, 0.4, 0.2],   # Brown
        1: [0.9, 0.9, 0.1],   # Yellow  
        2: [0.2, 0.8, 0.2],   # Green
        3: [0.1, 0.1, 0.7],   # Blue
    }
    
    # Sample subset for performance (every nth voxel)
    n_sample = min(10000, len(z_idx))  # Limit to 10k voxels
    indices = np.random.choice(len(z_idx), n_sample, replace=False)
    
    print(f"Creating mesh for {n_sample} voxels...")
    
    for i in indices:
        # Create cube mesh for this voxel
        cube = o3d.geometry.TriangleMesh.create_box(vs, vs, vs)
        
        # Position the cube
        x_pos = origin_x + x_idx[i] * vs
        y_pos = origin_y + y_idx[i] * vs
        z_pos = origin_z + z_idx[i] * vs
        cube.translate([x_pos, y_pos, z_pos])
        
        # Color the cube based on label
        label = labels[i]
        if label in color_map:
            color = color_map[label]
        else:
            color = [0.5, 0.5, 0.5]
        
        cube.paint_uniform_color(color)
        mesh_list.append(cube)
    
    # Combine all meshes
    combined_mesh = mesh_list[0]
    for mesh in mesh_list[1:]:
        combined_mesh += mesh
    
    # Save mesh
    mesh_file = "voxel_occupancy_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_file, combined_mesh)
    print(f"✓ Saved mesh: {mesh_file}")
    
    return combined_mesh

def create_height_slices():
    """Create 2D slices at different heights for visualization"""
    
    print("Creating height slice point clouds...")
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    vs = voxel_size['z']
    origin_z = origin['z']
    
    # Define height slices
    height_slices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # meters
    
    for height_m in height_slices:
        # Find corresponding Z index
        z_idx = int((height_m - origin_z) / vs)
        if z_idx < 0 or z_idx >= occupancy.shape[0]:
            continue
            
        # Get 2D slice
        slice_occ = occupancy[z_idx, :, :]
        slice_labels = dominant_labels[z_idx, :, :]
        
        # Find occupied pixels in this slice
        y_idx, x_idx = np.where(slice_occ > 0)
        
        if len(y_idx) == 0:
            continue
            
        # Convert to world coordinates
        world_x = origin['x'] + x_idx * voxel_size['x'] + voxel_size['x']/2
        world_y = origin['y'] + y_idx * voxel_size['y'] + voxel_size['y']/2
        world_z = np.full(len(y_idx), height_m)  # All at same height
        
        # Get labels
        labels = slice_labels[y_idx, x_idx]
        
        # Create colors
        colors = np.zeros((len(labels), 3))
        color_map = {
            0: [0.8, 0.4, 0.2],   # Brown
            1: [0.9, 0.9, 0.1],   # Yellow
            2: [0.2, 0.8, 0.2],   # Green
            3: [0.1, 0.1, 0.7],   # Blue
        }
        
        for i, label in enumerate(labels):
            if label in color_map:
                colors[i] = color_map[label]
            else:
                colors[i] = [0.5, 0.5, 0.5]
        
        # Create point cloud
        points = np.column_stack([world_x, world_y, world_z])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save slice
        slice_file = f"height_slice_{height_m:.1f}m.ply"
        o3d.io.write_point_cloud(slice_file, pcd)
        print(f"✓ Saved height slice: {slice_file}")

def create_las_output():
    """Create LAS file from voxel centers (if laspy available)"""
    
    try:
        import laspy
    except ImportError:
        print("laspy not available, skipping LAS output")
        return
        
    print("Creating LAS file...")
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    point_counts = data['point_counts']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    # Get occupied voxels
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    
    # Convert to world coordinates
    world_x = origin['x'] + x_idx * voxel_size['x'] + voxel_size['x']/2
    world_y = origin['y'] + y_idx * voxel_size['y'] + voxel_size['y']/2
    world_z = origin['z'] + z_idx * voxel_size['z'] + voxel_size['z']/2
    
    # Get attributes
    labels = dominant_labels[z_idx, y_idx, x_idx]
    counts = point_counts[z_idx, y_idx, x_idx]
    
    # Create LAS file
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = world_x
    las.y = world_y
    las.z = world_z
    
    # Add classification based on labels
    las.classification = labels.astype(np.uint8)
    
    # Add intensity based on point counts
    las.intensity = np.clip(counts * 1000, 0, 65535).astype(np.uint16)
    
    # Add colors
    colors = np.zeros((len(labels), 3), dtype=np.uint16)
    color_map = {
        0: [52428, 26214, 13107],   # Brown (65535 scale)
        1: [58982, 58982, 6553],    # Yellow
        2: [13107, 52428, 13107],   # Green  
        3: [6553, 6553, 45875],     # Blue
    }
    
    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = color_map[label]
        else:
            colors[i] = [32768, 32768, 32768]  # Gray
    
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]
    
    las_file = "voxel_occupancy.las"
    las.write(las_file)
    print(f"✓ Saved LAS file: {las_file}")

def print_visualization_guide():
    """Print guide for loading files in different software"""
    
    print("\n" + "="*60)
    print("3D VISUALIZATION GUIDE")
    print("="*60)
    
    print("\n🔷 CloudCompare:")
    print("- Load: voxel_occupancy_pointcloud.ply")
    print("- Load: voxel_occupancy_mesh.ply")
    print("- Load: height_slice_*.ply")
    print("- Colors are embedded in PLY files")
    
    print("\n🔷 MeshLab:")
    print("- Import: voxel_occupancy_pointcloud.ply")
    print("- Import: voxel_occupancy_mesh.ply")
    print("- Use Filters > Color > Per Vertex Color")
    
    print("\n🔷 QGIS (3D View):")
    print("- Load: voxel_occupancy_pointcloud.csv as delimited text")
    print("- Set X, Y, Z columns")
    print("- Use RGB values for coloring")
    
    print("\n🔷 ParaView:")
    print("- Open: voxel_occupancy_pointcloud.csv")
    print("- Use Table to Points filter")
    print("- Color by Label field")
    
    print("\n🔷 Blender:")
    print("- Import: voxel_occupancy_mesh.ply")
    print("- Use vertex colors for material")
    
    print("\n📊 File Summary:")
    print("- Point clouds: ~334K points")
    print("- Mesh: ~10K voxel cubes")
    print("- Height slices: 6 separate files")
    print("- Colors: Label-based (Brown/Yellow/Green/Blue)")

def main():
    """Main function to create all visualization outputs"""
    
    print("Converting voxel occupancy to 3D visualization formats...")
    print("This may take a few minutes...\n")
    
    # Create point cloud
    pcd, df = convert_voxels_to_pointcloud()
    
    # Create mesh (sampled for performance)
    create_voxel_mesh()
    
    # Create height slices
    create_height_slices()
    
    # Create LAS file if possible
    create_las_output()
    
    # Print visualization guide
    print_visualization_guide()
    
    print("\n✓ Conversion complete!")
    print("✓ Files ready for CloudCompare, MeshLab, QGIS, etc.")

if __name__ == "__main__":
    main()