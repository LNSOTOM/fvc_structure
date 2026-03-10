#%%
"""
Visualize 3D NIfTI feature and label volumes.

Supports two visualization modes:
1️⃣ Interactive 3D/2D Viewer (Napari)
    - pip install "napari[all]"
2️⃣ Static 2D Slice (Matplotlib)
    - pip install matplotlib
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try importing napari
try:
    import napari
except ImportError:
    napari = None
    print("Napari not found. Install with: pip install 'napari[all]'")

# -------------------------------------------------------------------
# 🧠 Function: Napari visualization
# -------------------------------------------------------------------
def view_with_napari(feature_path, label_path):
    """Open interactive 3D/2D viewer for feature and label volumes."""
    if napari is None:
        print("Napari not installed.  pip install 'napari[all]'")
        return

    print(f"\nOpening in Napari:")
    print(f"  Features → {feature_path}")
    print(f"  Labels   → {label_path}")

    # --- Load NIfTI volumes ---
    feature_img = nib.load(feature_path)
    label_img = nib.load(label_path)

    feature_data = feature_img.get_fdata()       # Shape (C, Z, Y, X)
    label_data = label_img.get_fdata().astype(np.uint8)  # Convert float → uint8

    # --- Normalize features for display ---
    print("Normalizing feature intensities for visibility...")
    feature_data_norm = np.zeros_like(feature_data, dtype=np.float32)
    for i in range(feature_data.shape[0]):
        band = feature_data[i]
        vmin, vmax = np.nanpercentile(band, (1, 99))
        feature_data_norm[i] = np.clip((band - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # --- Create viewer ---
    viewer = napari.Viewer(ndisplay=3)

    # --- Define band names and colors ---
    band_names = [
        "Band_1", "Band_2", "Band_3", "Band_4", "Band_5",
        "Max_Z", "PointCount"
    ]
    colormaps = ["gray", "magenta", "green", "cyan", "yellow", "orange", "blue"]

    # --- Add features (multi-channel volume) ---
    print("Adding feature layers...")
    image_layers = viewer.add_image(
        feature_data_norm,
        channel_axis=0,
        name=band_names[:feature_data.shape[0]],
        blending="additive",
        colormap=colormaps[:feature_data.shape[0]],
        contrast_limits=(0, 1),
        rendering="mip",  # Maximum Intensity Projection for 3D visibility
        attenuation=0.01,
    )

    # --- Initially show structure features only (bands 6 & 7) ---
    if isinstance(image_layers, list):
        for i, layer in enumerate(image_layers):
            if i < 5:
                layer.visible = False  # hide multispectral
            else:
                layer.visible = True
                layer.contrast_limits = (0.0, 0.5)
    else:
        print("ℹ️ Napari version may not return a list of layers, all bands visible.")

    # --- Add labels ---
    print("Adding label layer...")
    label_layer = viewer.add_labels(label_data, name="GroundTruth")

    # Custom label colors (if supported)
    try:
        label_layer.color = {
            65: 'yellow',   # Woody debris
            66: 'green',    # Grass
            67: 'orange',   # Shrub
            68: 'red',      # Mallee
        }
    except Exception:
        print("ℹ️ Custom label colors not supported in this Napari version.")

    # --- Adjust camera ---
    viewer.camera.zoom = 1.2
    viewer.dims.ndisplay = 3
    viewer.dims.current_step = (feature_data.shape[1] // 2, 0, 0)

    # --- Print summary ---
    print(f"\n✅ Added {feature_data.shape[0]} feature channels")
    print(f"🧊 Volume shape: (Z={feature_data.shape[1]}, Y={feature_data.shape[2]}, X={feature_data.shape[3]})")
    print(f"🏷️ Label dtype: {label_data.dtype}, unique values: {np.unique(label_data)}")
    print("🔹 Napari viewer open — press '2' for 2D, '3' for 3D, and toggle visibility in the left panel.\n")

    napari.run()


# -------------------------------------------------------------------
# 🖼️ Function: Matplotlib visualization
# -------------------------------------------------------------------
def plot_slice_with_matplotlib(feature_path, label_path, slice_z_index):
    """Plot a 2D slice of the feature and label volumes using Matplotlib."""
    print(f"\n--- Plotting slice {slice_z_index} with Matplotlib ---")

    feature_img = nib.load(feature_path)
    label_img = nib.load(label_path)

    feature_data = feature_img.get_fdata()
    label_data = label_img.get_fdata()

    # Ensure valid slice index
    if slice_z_index >= label_data.shape[0]:
        slice_z_index = label_data.shape[0] // 2
        print(f"Slice index out of bounds. Defaulting to middle slice: {slice_z_index}")

    label_slice = label_data[slice_z_index, :, :]
    feature_slice = feature_data[5, slice_z_index, :, :]  # Example: Max_Z channel

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # --- Feature slice ---
    ax = axes[0]
    im = ax.imshow(feature_slice, cmap='gray')
    ax.set_title(f"Feature (Channel 5: Max_Z) @ Z={slice_z_index}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Overlay labels ---
    ax = axes[1]
    masked_labels = np.ma.masked_where(label_slice == 0, label_slice)
    im = ax.imshow(feature_slice, cmap='gray')
    im_label = ax.imshow(masked_labels, cmap='viridis', interpolation='none', alpha=0.7)
    ax.set_title(f"Labels @ Z={slice_z_index}")
    ax.axis('off')
    plt.colorbar(im_label, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("nifti_slice_comparison.png")
    print("✅ Saved plot to nifti_slice_comparison.png")


# -------------------------------------------------------------------
# 🚀 Main Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    TID = "109"
    INPUT_DIR = Path("/media/laura/laura_usb/code/fvc_structure/phase_4_3d_cnn_training/nnUNet_input_data")

    feature_file = INPUT_DIR / f"{TID}_0000.nii.gz"
    label_file = INPUT_DIR / f"{TID}.nii.gz"

    if not feature_file.exists() or not label_file.exists():
        print(f"❌ ERROR: Missing NIfTI files in {INPUT_DIR}")
        print(f"Expected: {feature_file.name} and {label_file.name}")
    else:
        # --- Launch Napari Viewer ---
        view_with_napari(feature_file, label_file)

        # --- Optional: Static slice plot ---
        try:
            label_img = nib.load(label_file)
            z_dim = label_img.shape[0]
            middle_slice = z_dim // 2
            has_label = np.any(label_img.get_fdata() > 0, axis=(1, 2))
            labeled_slices = np.where(has_label)[0]
            if len(labeled_slices) > 0:
                middle_labeled_slice = labeled_slices[len(labeled_slices) // 2]
            else:
                middle_labeled_slice = middle_slice
            plot_slice_with_matplotlib(feature_file, label_file, middle_labeled_slice)
        except Exception as e:
            print(f"⚠️ Could not plot with Matplotlib: {e}")

# %%
### test 2
#%%
"""
Visualize 3D NIfTI feature and label volumes (float32 bands).

Modes:
1️⃣ Napari (3D/2D Interactive Viewer)
    - pip install "napari[all]"
2️⃣ Matplotlib (Static 2D Slice)
    - Shows all bands in float32 and overlay labels
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import napari
except ImportError:
    napari = None
    print("Napari not found. Install with: pip install 'napari[all]'")

# -------------------------------------------------------------------
# 🧠 Napari Visualization
# -------------------------------------------------------------------
def view_with_napari(feature_path, label_path):
    """Open interactive 3D/2D viewer for feature and label volumes."""
    if napari is None:
        print("Napari not installed. pip install 'napari[all]'")
        return

    print(f"\nOpening in Napari:")
    print(f"  Features → {feature_path}")
    print(f"  Labels   → {label_path}")

    # Load NIfTI volumes
    feature_img = nib.load(feature_path)
    label_img = nib.load(label_path)
    feature_data = feature_img.get_fdata().astype(np.float32)
    label_data = label_img.get_fdata().astype(np.uint8)

    # Normalize each band for visibility
    print("Normalizing feature intensities for visibility...")
    feature_data_norm = np.zeros_like(feature_data, dtype=np.float32)
    for i in range(feature_data.shape[0]):
        band = feature_data[i]
        vmin, vmax = np.nanpercentile(band, (1, 99))
        feature_data_norm[i] = np.clip((band - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # Create viewer
    viewer = napari.Viewer(ndisplay=3)

    band_names = [
        "Band_1", "Band_2", "Band_3", "Band_4", "Band_5",
        "Max_Z", "PointCount"
    ]
    colormaps = ["gray", "magenta", "green", "cyan", "yellow", "orange", "blue"]

    # Add features
    image_layers = viewer.add_image(
        feature_data_norm,
        channel_axis=0,
        name=band_names[:feature_data.shape[0]],
        blending="additive",
        colormap=colormaps[:feature_data.shape[0]],
        contrast_limits=(0, 1),
        rendering="mip",
        attenuation=0.01,
    )

    # Show structure features only (bands 6–7)
    if isinstance(image_layers, list):
        for i, layer in enumerate(image_layers):
            if i < 5:
                layer.visible = False
            else:
                layer.visible = True
                layer.contrast_limits = (0.0, 0.5)

    # Add label layer
    label_layer = viewer.add_labels(label_data, name="GroundTruth")
    try:
        label_layer.color = {
            65: 'yellow',
            66: 'green',
            67: 'orange',
            68: 'red',
        }
    except Exception:
        print("ℹ️ Custom label colors not supported in this Napari version.")

    viewer.camera.zoom = 1.2
    viewer.dims.ndisplay = 3
    viewer.dims.current_step = (feature_data.shape[1] // 2, 0, 0)

    print(f"\n✅ Added {feature_data.shape[0]} feature channels")
    print(f"🧊 Volume shape: (Z={feature_data.shape[1]}, Y={feature_data.shape[2]}, X={feature_data.shape[3]})")
    print(f"🏷️ Label dtype: {label_data.dtype}, unique values: {np.unique(label_data)}")
    print("🔹 Napari viewer open — press '2' for 2D, '3' for 3D.\n")

    napari.run()


# -------------------------------------------------------------------
# 🖼️ Matplotlib Visualization — all 7 bands as float32
# -------------------------------------------------------------------
def plot_slice_with_matplotlib(feature_path, label_path, slice_z_index):
    """Plot all 7 bands as float32 and overlay labels."""
    print(f"\n--- Plotting slice {slice_z_index} with Matplotlib ---")

    feature_img = nib.load(feature_path)
    label_img = nib.load(label_path)

    feature_data = feature_img.get_fdata().astype(np.float32)
    label_data = label_img.get_fdata().astype(np.uint8)

    if slice_z_index >= label_data.shape[0]:
        slice_z_index = label_data.shape[0] // 2
        print(f"Slice index out of bounds. Using middle slice {slice_z_index}")

    label_slice = label_data[slice_z_index, :, :]

    band_names = [
        "Band_1", "Band_2", "Band_3", "Band_4", "Band_5",
        "Max_Z", "PointCount"
    ]

    n_bands = feature_data.shape[0]
    ncols = 4
    nrows = int(np.ceil((n_bands + 1) / ncols))  # +1 for label overlay

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    axes = axes.flatten()

    # Plot each band as float32 image
    for i in range(n_bands):
        ax = axes[i]
        band_slice = feature_data[i, slice_z_index, :, :]
        im = ax.imshow(band_slice, cmap='gray')
        ax.set_title(f"{band_names[i]} (float32)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overlay labels on the last subplot
    ax = axes[n_bands]
    base_band = feature_data[5, slice_z_index, :, :]  # Max_Z as background
    masked_labels = np.ma.masked_where(label_slice == 0, label_slice)
    im_base = ax.imshow(base_band, cmap='gray')
    im_label = ax.imshow(masked_labels, cmap='viridis', interpolation='none', alpha=0.7)
    ax.set_title("Label Overlay")
    ax.axis('off')
    plt.colorbar(im_label, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for j in range(n_bands + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("nifti_allbands_float32.png", dpi=300)
    print("✅ Saved all-band visualization to nifti_allbands_float32.png")


# -------------------------------------------------------------------
# 🚀 Main Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    TID = "109"
    INPUT_DIR = Path("/media/laura/laura_usb/code/fvc_structure/phase_4_3d_cnn_training/nnUNet_input_data")

    feature_file = INPUT_DIR / f"{TID}_0000.nii.gz"
    label_file = INPUT_DIR / f"{TID}.nii.gz"

    if not feature_file.exists() or not label_file.exists():
        print(f"❌ ERROR: Missing NIfTI files in {INPUT_DIR}")
        print(f"Expected: {feature_file.name} and {label_file.name}")
    else:
        # --- Launch Napari Viewer ---
        view_with_napari(feature_file, label_file)

        # --- Generate all-band float32 static visualization ---
        try:
            label_img = nib.load(label_file)
            z_dim = label_img.shape[0]
            middle_slice = z_dim // 2
            has_label = np.any(label_img.get_fdata() > 0, axis=(1, 2))
            labeled_slices = np.where(has_label)[0]
            if len(labeled_slices) > 0:
                middle_labeled_slice = labeled_slices[len(labeled_slices) // 2]
            else:
                middle_labeled_slice = middle_slice
            plot_slice_with_matplotlib(feature_file, label_file, middle_labeled_slice)
        except Exception as e:
            print(f"⚠️ Could not plot with Matplotlib: {e}")

# %%
