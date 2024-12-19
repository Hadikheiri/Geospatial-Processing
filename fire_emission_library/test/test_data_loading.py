from fire_emission_analysis.data_loading import load_shapefile, load_raster, save_raster
from fire_emission_analysis.fire_severity import calculate_fire_severity
from fire_emission_analysis.land_cover import classify_land_cover
from fire_emission_analysis.emission_calculation import calculate_emissions
from fire_emission_analysis.pollution_analysis import detect_smoke_extent, generate_pollution_heatmap
import numpy as np

# Paths to input data
shapefile_path = "input/fire_boundary.shp"
raster_path = "input/ndvi_before_fire.tif"
raster_path_after = "input/ndvi_after_fire.tif"

# Load data
shapefile = load_shapefile(shapefile_path)
before_meta, before_data = load_raster(raster_path)
after_meta, after_data = load_raster(raster_path_after)

# Calculate NDVI difference (fire severity proxy)
ndvi_diff = before_data - after_data
severity_thresholds = {1: -0.2, 2: -0.5, 3: -0.8}  # Adjust thresholds based on your context
fire_severity_map = calculate_fire_severity(ndvi_diff, severity_thresholds)

# Save the fire severity map
save_raster(fire_severity_map, before_meta, "output/fire_severity_map.tif")

# Land cover classification
# Example dummy data for land cover classification
features = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])  # Training features
labels = ["forest", "grassland", "urban"]
region_features = np.array([[0.2, 0.3], [0.7, 0.5], [0.8, 0.9]])  # Region features to classify
classified_map = classify_land_cover(features, labels, region_features)

# Example emission factors (adjust for real data)
emission_factors = {"forest": 1.2, "grassland": 0.8, "urban": 0.5}
classified_area = {"forest": 10, "grassland": 5, "urban": 2}  # Example areas (dummy)
emissions = calculate_emissions(classified_area, emission_factors)
print(f"Estimated CO2 emissions: {emissions} tons")

# Smoke and pollution analysis
smoke_index = after_data - before_data  # Example proxy for smoke index
smoke_extent = detect_smoke_extent(smoke_index, threshold=0.1)
heatmap = generate_pollution_heatmap(smoke_index)

# Save heatmap
save_raster(heatmap, before_meta, "output/pollution_heatmap.tif")
print(f"Smoke extent (pixels): {smoke_extent}")
print("Processing complete. Outputs saved in the output/ folder.")
