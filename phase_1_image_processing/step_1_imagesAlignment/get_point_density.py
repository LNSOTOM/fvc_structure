
#%%
import laspy

# === Load LiDAR file ===
las = laspy.read("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")

# === Extract coordinates ===
x, y, z = las.x, las.y, las.z

# === Calculate point density ===
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
area_m2 = (xmax - xmin) * (ymax - ymin)
density = len(x) / area_m2
recommended_res = (1 / density) ** 0.5
print(f"Point density: {density:.2f} pts/m² → Recommended resolution: {recommended_res:.4f} m")


# %%
