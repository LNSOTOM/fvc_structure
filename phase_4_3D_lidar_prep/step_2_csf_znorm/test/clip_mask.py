#%%
import json
from pathlib import Path

import laspy
import numpy as np
import pdal
from scipy.spatial import KDTree
from tqdm import tqdm


def apply_mask_override(input_las: Path, output_las: Path, mask_tif: Path) -> int:
    """Stage mask classes inside the LAS file using PDAL colorization."""
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {"type": "filters.colorization", "raster": str(mask_tif), "dimensions": "MaskClass:1"},
            {"type": "writers.las", "filename": str(output_las), "extra_dims": "all"},
        ]
    }

    print("\n--- EXECUTING STEP 2, PART 1: Staging Mask Values ---")
    pipeline_runner = pdal.Pipeline(json.dumps(pipeline))
    point_count = pipeline_runner.execute()
    print(f"✅ Part 1 complete. Created intermediate file with {point_count} points.")
    print(f"   Saved to: {output_las}")
    return point_count


def expand_classification_via_ground_points(
    input_las: Path,
    output_las: Path,
    search_radius: float = 0.35,
) -> None:
    """Expand mask classifications to nearby points using a spatial search."""

    print("\n--- EXECUTING STEP 2, PART 2: Expanding Classification ---")
    with laspy.open(input_las) as reader:
        try:
            _ = reader.header
        except OSError as exc:
            raise RuntimeError(f"Unable to read LAS header from {input_las}") from exc

        progress = tqdm(total=reader.header.point_count, desc="Expanding Classes", unit="pts")

        with laspy.open(output_las, mode="w", header=reader.header) as writer:
            for points in reader.chunk_iterator(2_000_000):
                if not hasattr(points, "MaskClass"):
                    raise AttributeError("MaskClass dimension missing. Ensure Part 1 ran successfully.")

                mask_values = points.MaskClass.astype(np.int16)
                mask_values[(mask_values < 1) | (mask_values > 4)] = 0
                mask_values = np.where(mask_values > 0, mask_values + 63, 0)  # 1→64, 2→65, etc.
                mask_valid = mask_values > 0

                points.classification[mask_valid] = mask_values[mask_valid]

                source_idx = np.where(mask_valid)[0]
                target_idx = np.where(~mask_valid)[0]

                if source_idx.size and target_idx.size:
                    source_xy = np.column_stack((points.x[source_idx], points.y[source_idx]))
                    tree = KDTree(source_xy)

                    target_xy = np.column_stack((points.x[target_idx], points.y[target_idx]))
                    distances, neighbour_idx = tree.query(
                        target_xy, k=1, distance_upper_bound=search_radius
                    )

                    valid = distances != np.inf
                    if np.any(valid):
                        target_valid_idx = target_idx[valid]
                        source_valid_idx = source_idx[neighbour_idx[valid]]
                        points.classification[target_valid_idx] = mask_values[source_valid_idx]

                writer.write_points(points)
                progress.update(len(points))

        progress.close()
    print("Classification expansion complete.")


if __name__ == "__main__":
    raster_name = "tiles_multispectral.26"
    base_dir = Path("./lidar_processing_steps_correct_order")
    base_dir.mkdir(exist_ok=True)

    input_ground_las = base_dir / f"{raster_name}_step1_ground_classified.las"
    mask_tif = Path(
        "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/"
        "annotation_raster/downsample_5cm_nearest/mask_structure_3072.26.tif"
    )

    intermediate_las = base_dir / f"{raster_name}_step2_authoritative_mask.las"
    output_hybrid_las = base_dir / f"{raster_name}_step2_hybrid_classified.las"

    if not input_ground_las.exists():
        raise FileNotFoundError(f"Input file from Step 1 not found: {input_ground_las}")
    if not mask_tif.exists():
        raise FileNotFoundError(f"Authoritative TIF mask not found: {mask_tif}")

    apply_mask_override(input_ground_las, intermediate_las, mask_tif)

    expand_classification_via_ground_points(
        input_las=intermediate_las,
        output_las=output_hybrid_las,
        search_radius=0.15,
    )

    if intermediate_las.exists():
        intermediate_las.unlink()

    print("\n--- ALL DONE ---")
    print(f"Final expanded classification file saved to: {output_hybrid_las}")
    print("\n--- NEXT STEP ---")
    print("Run 'step_3_normalize_height.py' to calculate height above ground.")


# %%
##test 2

# --- User Configuration ---
# You MUST change these file paths to match your data.

# This is your main file with the initial ground classification (e.g., from PIC 1)
FILE_TO_CORRECT = "/media/laura/laura_usb/code/fvc_structure/phase_3_3D_lidar_prep/step_2_csf/lidar_processing_steps_correct_order/tiles_multispectral.26_step1_ground_classified_conservative.las"

# This is your second file that has the accurate vegetation labels
LABEL_FILE = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raw_z/tiles_multispectral.26_raw_z.las"

# This will be the name of the new, corrected output file
OUTPUT_FILE = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raw_z/tiles_multispectral.26_raw_z_new.las"
# --- Reclassification Logic ---

# The class code for "Ground" in your primary file that we want to check.
# According to ASPRS standard, this is usually 2.
GROUND_CLASS_TO_CHECK = 2

# A list of class codes in your LABEL_FILE that represent vegetation.
# The script will transfer these labels if it finds them.
# Example: [3, 4, 5] for low, medium, and high vegetation.
VEGETATION_CLASSES_TO_USE = [65, 66, 67, 68]

# Maximum distance (in LAS units) to search for a matching label point
NEIGHBOR_MATCH_RADIUS = 0.15

# --- Main Script ---
def main():
    """
    Corrects a ground classification LAS/LAZ file by transferring
    classification labels from a second source file.
    """
    try:
        # 1. Read both the file to be corrected and the label file
        print(f"Reading file to correct: {FILE_TO_CORRECT}")
        with laspy.open(FILE_TO_CORRECT) as f:
            las_to_correct = f.read()

        print(f"Reading label file: {LABEL_FILE}")
        with laspy.open(LABEL_FILE) as f:
            las_labels = f.read()

        if len(las_to_correct.points) != len(las_labels.points):
            print("⚠️  Warning: Files have different point counts. Using spatial nearest-neighbour matching.")
        else:
            print(f"Files successfully read. Both contain {len(las_to_correct.points)} points.")

        ground_indices = np.where(las_to_correct.classification == GROUND_CLASS_TO_CHECK)[0]
        if ground_indices.size == 0:
            print("No ground-classified points found in the source file; nothing to correct.")
            return

        label_mask = np.isin(las_labels.classification, VEGETATION_CLASSES_TO_USE)
        if label_mask.sum() == 0:
            print("No vegetation-labelled points found in the label file. Aborting correction.")
            return

        label_coords = np.column_stack(
            (las_labels.x[label_mask], las_labels.y[label_mask], las_labels.z[label_mask])
        )
        label_classes = las_labels.classification[label_mask]
        tree = KDTree(label_coords)

        ground_coords = np.column_stack(
            (las_to_correct.x[ground_indices], las_to_correct.y[ground_indices], las_to_correct.z[ground_indices])
        )
        distances, neighbour_idx = tree.query(
            ground_coords, k=1, distance_upper_bound=NEIGHBOR_MATCH_RADIUS
        )

        valid_matches = np.isfinite(distances)
        matched_count = int(np.count_nonzero(valid_matches))

        if matched_count == 0:
            print(
                f"No ground points found a vegetation neighbour within {NEIGHBOR_MATCH_RADIUS} units; no changes applied."
            )
        else:
            matched_ground_indices = ground_indices[valid_matches]
            matched_label_indices = neighbour_idx[valid_matches]
            new_classification = las_to_correct.classification.copy()
            new_classification[matched_ground_indices] = label_classes[matched_label_indices]
            las_to_correct.classification = new_classification
            print(
                f"Applied vegetation labels to {matched_count} ground points out of {ground_indices.size} candidates."
            )

        unmatched = ground_indices.size - matched_count
        if unmatched > 0:
            print(
                f"{unmatched} ground points had no vegetation neighbour within the search radius and were left unchanged."
            )

        # 5. Write the fully corrected point cloud to a new file
        print(f"Writing corrected point cloud to {OUTPUT_FILE}...")
        with laspy.open(OUTPUT_FILE, mode="w", header=las_to_correct.header) as out_las:
            out_las.write_points(las_to_correct.points)

        print("Processing complete! Your corrected file is ready.")

    except FileNotFoundError as e:
        print(f"Error: A file was not found. Details: {e}")
        print("Please ensure both input file paths are correct and the files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

# %%
##test 3 (legacy variant retained for reference)
import numpy as np
import laspy
from scipy.spatial import cKDTree

# --- User Configuration ---
# File paths are set based on your previous execution.

# This is your main file with the initial ground classification to be corrected.
FILE_TO_CORRECT = "/media/laura/laura_usb/code/fvc_structure/phase_3_3D_lidar_prep/step_2_csf/lidar_processing_steps_correct_order/tiles_multispectral.26_step1_ground_classified_conservative.las"

# This is your second file that has the accurate "truth" labels.
LABEL_FILE = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raw_z/tiles_multispectral.26_raw_z.las"

# This will be the name of the new, corrected output file.
OUTPUT_FILE = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raw_z/tiles_multispectral.26_final_corrected.las"

# --- Reclassification Logic ---
# The class code for "Ground" that we will be checking and potentially changing.
# According to ASPRS standard, this is usually 2.
GROUND_CLASS_CODE = 2

# A small tolerance for matching points, to account for tiny floating point differences.
MATCHING_TOLERANCE = 1e-6

# --- Main Script ---
def main():
    """
    Corrects a ground classification LAS/LAZ file by spatially matching points
    to a label file and transferring classifications for misclassified ground points.
    """
    try:
        # 1. Read both the file to be corrected and the label file
        print(f"Reading file to correct: {FILE_TO_CORRECT}")
        with laspy.open(FILE_TO_CORRECT) as f:
            las_to_correct = f.read()

        print(f"Reading label file: {LABEL_FILE}")
        with laspy.open(LABEL_FILE) as f:
            las_labels = f.read()
        print("Files successfully read.")

        # 2. Build a KD-Tree from the label file's points for fast spatial searching.
        label_points_xyz = np.vstack((las_labels.x, las_labels.y, las_labels.z)).transpose()
        print("Building a spatial index (KD-Tree) from the label file...")
        kdtree = cKDTree(label_points_xyz)
        print("Spatial index created.")

        # 3. Get the indices and coordinates of only the ground points from the source file.
        ground_indices_in_source = np.where(las_to_correct.classification == GROUND_CLASS_CODE)[0]
        
        if len(ground_indices_in_source) == 0:
            print("No points classified as ground were found. No changes will be made.")
            return

        print(f"Found {len(ground_indices_in_source)} ground points to check for potential correction.")
        source_points_xyz = np.vstack((las_to_correct.x, las_to_correct.y, las_to_correct.z)).transpose()
        ground_points_to_query = source_points_xyz[ground_indices_in_source]

        # 4. Find the corresponding point in the label file for each ground point.
        distances, indices_in_labels = kdtree.query(ground_points_to_query, k=1)
        
        # 5. Determine which points to change based on your specific logic.
        # Find points that are perfect spatial matches.
        perfect_matches_mask = distances < MATCHING_TOLERANCE
        
        # Of these matches, get their classifications from the label file.
        matched_labels = las_labels.classification[indices_in_labels[perfect_matches_mask]]
        
        # Find which of these labels are NOT ground (i.e., they are misclassified).
        misclassified_mask = matched_labels != GROUND_CLASS_CODE
        
        # Get the final indices in the *original* source file that need to be corrected.
        final_indices_to_correct = ground_indices_in_source[perfect_matches_mask][misclassified_mask]
        
        # Get the corresponding *new* classification values from the label file.
        labels_to_transfer = matched_labels[misclassified_mask]

        num_to_correct = len(final_indices_to_correct)
        if num_to_correct > 0:
            print(f"Found {num_to_correct} points that were incorrectly classified as Ground. Correcting them now...")
            
            # 6. Apply the correction to a copy of the original classification data.
            new_classification = las_to_correct.classification.copy()
            new_classification[final_indices_to_correct] = labels_to_transfer
            las_to_correct.classification = new_classification
        else:
            print("No misclassified ground points were found based on the label file. The data is already correct.")

        # 7. Write the fully corrected point cloud to a new file.
        print(f"Writing corrected point cloud to {OUTPUT_FILE}...")
        las_to_correct.write(OUTPUT_FILE)
        print("Processing complete! Your final corrected file is ready.")

    except FileNotFoundError as e:
        print(f"Error: A file was not found. Please double-check your file paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

# %%
