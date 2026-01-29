
#%%
import rasterio
import os


# Input and output folders
#site1 - supersite [MEDIUM]
##3072x3072
input_folder = '/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw'
output_folder_5b = '/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/'


##site4 -  Vegetation [LOW] (potentially site DD0013)
# input_folder = '/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw'
# output_folder_5b = '/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/'


# Check for directory existence, if not, create
os.makedirs(output_folder_5b, exist_ok=True)
# os.makedirs(output_folder_5b, exist_ok=True)

# Bands to save for 10-band files
bands_to_save_10b = [2, 4, 6, 8, 10]

# List all files in the input folder
file_list = os.listdir(input_folder)

for input_filename in file_list:
    try:
        input_path = os.path.join(input_folder, input_filename)

        with rasterio.open(input_path) as src:
            # Error check if raster file is not in expected format
            if src.count < len(bands_to_save_10b) or src.count < 5:
                print(f"Skipping {input_filename}: It does not meet band selection criteria.")
                continue
            
            # Storing selected and reduced bands using in-memory slicing
            profile = src.profile
            # Read the data for 10b output
            data_10b = [src.read(b) for b in bands_to_save_10b]
            # Use pre-read file for 5b operation to pick starting bands based on present count
            start_5b_bands = list(range(1, min(6, src.count + 1)))
            data_5b = [src.read(i) for i in start_5b_bands]

        # Creating 10-band stacked Raster
        profile.update(count=len(data_10b))
        output_path_10b = os.path.join(output_folder_5b, f"{input_filename}")
        with rasterio.open(output_path_10b, 'w', **profile) as dst:
            for idx, data in enumerate(data_10b, start=1):
                dst.write(data, idx)
                

    except Exception as e:
        print(f"Error occurred with {input_filename}: {e}")

print("Raster band reduction and saving are complete.")
# %%
