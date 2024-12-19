### fire_emission_analysis/data_loading.py
import geopandas as gpd
import rasterio

def load_shapefile(shapefile_path):
    """Load a shapefile and return a GeoDataFrame."""
    shapefile = gpd.read_file(shapefile_path)
    return shapefile

def load_raster(raster_path):
    """Load a raster and return metadata and data."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        return src.meta, data

def save_raster(data, meta, output_path):
    """Save a raster file."""
    with rasterio.open(output_path, "w", **meta) as dest:
        dest.write(data, 1)
