# Import required scientific and data processing libraries
import scipy  # Scientific computing library for advanced mathematical operations
import numpy as np  # Fundamental package for numerical computations
import openeo  # Library for connecting to and processing Earth observation data
from openeo.extra.spectral_indices import compute_indices  # Module for calculating vegetation indices

# Import visualization libraries
import matplotlib.pyplot as plt  # Main plotting library
import matplotlib  # Base matplotlib library for customization
import matplotlib.patches as mpatches  # For creating plot legends and shapes
import rasterio  # Library for reading and writing geospatial raster data
from rasterio.plot import show  # Function for displaying raster data

# Import mapping and interactive visualization libraries
from ipyleaflet import Map, DrawControl, TileLayer  # Interactive mapping components
from ipywidgets import Layout  # Widget layout management for Jupyter

# Import date handling libraries
from datetime import datetime  # Basic date and time functionality
from dateutil.relativedelta import relativedelta  # Advanced date arithmetic

# Import additional visualization and machine learning libraries
import matplotlib.colors as mcolors  # Color handling utilities
from sklearn.cluster import KMeans  # K-means clustering algorithm
from sklearn.preprocessing import StandardScaler  # Data standardization
from sklearn.impute import SimpleImputer  # For handling missing values
from scipy.ndimage import median_filter  # For noise reduction in images

def connect_to_openeo():
    """
    Establish a connection to the OpenEO API and authenticate.
    Returns:
        openeo.Connection: An authenticated connection object.
    """
    try:
        # Connect to the OpenEO API and authenticate using OIDC
        connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
        print("Successfully connected and authenticated with OpenEO.")
        return connection
    except Exception as e:
        print(f"Failed to connect to OpenEO: {e}")
        raise

def load_map():
    """
    Creates and configures an interactive map with drawing capabilities.
    Returns:
        Map: An ipyleaflet Map object with drawing controls
    """
    # Define the map center (latitude, longitude) - Currently set to Milan, Italy
    center_location = [45.4642, 9.1900]

    # Create a map object with specified dimensions
    m = Map(center=center_location, zoom=5, layout=Layout(height="600px"))

    # Add OpenStreetMap as the base layer
    osm_layer = TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", name="OpenStreetMap")
    m.add_layer(osm_layer)

    # Initialize drawing control for user interactions
    draw_control = DrawControl()

    # Configure rectangle drawing options with blue color scheme
    draw_control.rectangle = {
        "shapeOptions": {
            "color": "#3388ff",
            "fillColor": "#3388ff",
            "fillOpacity": 0.5,
        }
    }

    # Disable other drawing tools to only allow rectangle drawing
    draw_control.circle = {}
    draw_control.circlemarker = {}
    draw_control.polygon = {}
    draw_control.polyline = {}
    draw_control.marker = {}

    # Define global variable to store the drawn extent
    extent = {}

    def handle_draw(self, action, geo_json):
        """
        Callback function to process drawn rectangles and extract their coordinates
        Args:
            action: The drawing action performed
            geo_json: The geometry data of the drawn shape
        """
        global extent  # Access the global extent variable

        # Process only polygon (rectangle) geometries
        if geo_json['geometry']['type'] == 'Polygon':
            coordinates = geo_json['geometry']['coordinates'][0]

            # Calculate bounding box coordinates
            west = min([coord[0] for coord in coordinates])
            south = min([coord[1] for coord in coordinates])
            east = max([coord[0] for coord in coordinates])
            north = max([coord[1] for coord in coordinates])

            # Store the extent coordinates
            extent = {"west": west, "south": south, "east": east, "north": north}
            print(f"Rectangle Extent: {extent}")

    # Connect the drawing handler to the control
    draw_control.on_draw(handle_draw)
    
    # Add drawing control to the map
    m.add_control(draw_control)

    # Return the configured map
    return m


# Global variable 'extent' stores the geographical bounds selected by the user
# Format: {"west": float, "south": float, "east": float, "north": float}

def getBAP(scl, data, reducer="first"):
    """
    Get Best Available Pixel by applying cloud and shadow masking.
    
    Args:
        scl: Scene classification layer
        data: Input satellite data
        reducer: Method to reduce temporal dimension (default: "first")
    
    Returns:
        Masked data cube with selected pixels
    """
    # Create mask for clouds (3), cloud shadows (8), and semi-transparent clouds (9, 10)
    mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)

    # Apply Gaussian smoothing to reduce noise in the mask
    # Create 2D Gaussian kernel
    g = scipy.signal.windows.gaussian(11, std=1.6)
    kernel = np.outer(g, g)
    kernel = kernel / kernel.sum()  # Normalize kernel

    # Apply kernel and threshold to create final mask
    mask = mask.apply_kernel(kernel)
    mask = mask > 0.1

    # Apply mask to input data
    data_masked = data.mask(mask)

    # Reduce temporal dimension using specified method
    return data_masked.reduce_dimension(reducer=reducer, dimension="t")

def get_date_input(prompt):
    """
    Get a valid date input from the user.
    
    Args:
        prompt: Text to display when asking for input
    
    Returns:
        datetime object representing the input date
    """
    while True:
        user_input = input(prompt)
        try:
            # Parse date string to datetime object
            date = datetime.strptime(user_input, "%Y-%m-d")
            return date
        except ValueError:
            # Handle invalid date format
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

def get_start_and_end_dates():
    """
    Get and validate start and end dates from user input.
    
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    # Get input dates
    start_date = get_date_input("Enter the start date (YYYY-MM-DD): ")
    end_date = get_date_input("Enter the end date (YYYY-MM-DD): ")

    # Validate date order
    if start_date > end_date:
        print("Start date cannot be after end date. Please try again.")
        return get_start_and_end_dates()  # Recursive call for new input
    else:
        return start_date, end_date

def load_pre_ndvi(connection, extent, start_date, end_date):
    """
    Load and process pre-event NDVI (Normalized Difference Vegetation Index) data.
    
    Args:
        connection: OpenEO connection object
        extent: Geographical bounds
        start_date: Start of time period
        end_date: End of time period
    """
    # Load Sentinel-2 L2A collection for pre-event period
    s2pre = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=["2022-04-01", "2022-08-30"],
        spatial_extent=extent,
        bands=["B04", "B08"],  # Bands needed for NDVI calculation
        max_cloud_cover=10,
    )

    # Create maximum composite over time period
    s2pre_max = s2pre.reduce_dimension(dimension="t", reducer="max")
    # Calculate NDVI and save to file
    ndvi_pre = s2pre_max.ndvi()
    ndvi_pre.download("NDVI_PRE.tiff")

def plot_pre_ndvi():
    """
    Create and display a visualization of pre-event NDVI data.
    """
    # Open NDVI file
    b = rasterio.open("NDVI_PRE.tiff")
    
    # Set up plot
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(18, 6))
    # Display NDVI data with specified color scheme
    im = show(b.read(1), vmin=0, vmax=1, transform=b.transform, ax=ax, cmap='Reds')
    
    # Add colorbar and labels
    cbar = plt.colorbar(im.get_images()[0], ax=ax, orientation='vertical')
    cbar.set_label('NDVI Value')
    ax.set_title("Pre Event NDVI")
    plt.show()

def load_post_ndvi(connection, extent, start_date, end_date):
    """
    Load and process post-event NDVI data.
    
    Args:
        connection: OpenEO connection object
        extent: Geographical bounds
        start_date: Start of time period
        end_date: End of time period
    """
    # Calculate end date for post-event period (2 months after event)
    final_date = end_date + relativedelta(months=+2)
    
    # Load Sentinel-2 L2A collection for post-event period
    s2post = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=[start_date, final_date],
        spatial_extent=extent,
        bands=["B04", "B08"],
        max_cloud_cover=10,
    )
    
    # Calculate and save NDVI
    ndvi_post = s2post.ndvi()
    ndvi_post.download("Post_NDVI.tiff")

def plot_post_ndvi():
    """
    Create and display a visualization of post-event NDVI data.
    """
    # Open NDVI file
    b = rasterio.open("Post_NDVI.tiff")
    
    # Set up plot
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(18, 6))
    # Display NDVI data with specified color scheme
    im = show(b.read(1), vmin=0, vmax=1, transform=b.transform, ax=ax, cmap='Reds')
    
    # Add colorbar and labels
    cbar = plt.colorbar(im.get_images()[0], ax=ax, orientation='vertical')
    cbar.set_label('NDVI Value')
    ax.set_title("Post Event NDVI")
    plt.show()


def fire_detector_ndvi():
    """
    Detects fire-affected areas by comparing pre and post-fire NDVI values.
    Creates and displays a visualization of the NDVI difference.
    """
    # Open both NDVI files using context managers for automatic cleanup
    with rasterio.open("NDVI_PRE.tiff") as src_pre, rasterio.open("Post_NDVI.tiff") as src_post:
        # Load NDVI data into numpy arrays
        ndvi_pre = src_pre.read(1)
        ndvi_post = src_post.read(1)

        # Calculate NDVI difference (negative values indicate vegetation loss)
        final_fire_NDVI = ndvi_post - ndvi_pre

        # Create visualization
        f, ax = plt.subplots(1, 1, dpi=100, figsize=(18, 6))
        # Display difference map with value range [-1, 1]
        im = ax.imshow(final_fire_NDVI, vmin=-1, vmax=1, cmap='Greens')

        # Add colorbar with label
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('NDVI Value')

        ax.set_title("Fire Area - NDVI Difference")
        plt.show()

def load_pre_nbr(connection, extent, start_date, end_date):
    """
    Loads and processes pre-fire NBR (Normalized Burn Ratio) and NDWI (Normalized Difference Water Index) data.
    
    Args:
        connection: OpenEO connection object
        extent: Geographical bounds
        start_date: Start of time period
        end_date: End of time period
    """
    # Load Sentinel-2 data with required bands
    s2pre = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=["2022-04-01", "2022-08-30"],
        spatial_extent=extent,
        bands=["B03","B04","B08","B12"],  # Bands needed for NBR and NDWI
        max_cloud_cover=10,
    )

    # Create temporal composite using maximum values
    s2pre_max = s2pre.reduce_dimension(dimension="t", reducer="max")

    # Calculate NDWI using Green (B03) and NIR (B08) bands
    # NDWI helps identify water bodies
    green = s2pre_max.band("B03")
    nir = s2pre_max.band("B08")
    ndwi = (green - nir) / (green + nir)

    # Create water mask (NDWI > 0 indicates potential water)
    threshold = 0.0
    water_mask = ndwi > threshold

    # Note: Commented code below shows how to mask out water areas from NDVI
    # ndvi_pr = ndvi_pre.mask(~water_mask)

    # Calculate NBR using NIR (B08) and SWIR (B12) bands
    SWIR = s2pre_max.band("B12")
    NBR_Pre = (nir - SWIR) / (nir + SWIR)

    # Save results to files
    NBR_Pre.download("NBR_PRE.tiff")
    ndwi.download("NDWI.tiff")

def post_nbr(connection, extent, start_date, end_date):
    """
    Loads and processes post-fire NBR (Normalized Burn Ratio) data.
    
    Args:
        connection: OpenEO connection object
        extent: Geographical bounds
        start_date: Start of time period
        end_date: End of time period
    """
    # Calculate end date for post-fire period
    final_date = end_date + relativedelta(months=+2)
    
    # Load post-fire Sentinel-2 data
    s2post = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=[start_date, final_date],
        spatial_extent=extent,
        bands=["B03","B04","B08","B12"],
        max_cloud_cover=10,
    )

    # Calculate post-fire NBR
    NIR = s2post.band("B08")
    SWIR = s2post.band("B12")
    NBR_Post = (NIR - SWIR) / (NIR + SWIR)
    NBR_Post.download("NBR_Post.tiff")

def plot_nbr():
    """
    Creates and displays a visualization of the post-fire NBR data.
    """
    # Load NBR data
    b = rasterio.open("NBR_Post.tiff")
    
    # Create visualization
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(18, 6))
    im = show(b.read(1), vmin=0, vmax=1, transform=b.transform, ax=ax, cmap='Reds')
    
    # Add colorbar with label
    cbar = plt.colorbar(im.get_images()[0], ax=ax, orientation='vertical')
    cbar.set_label('NBR Value')
    
    ax.set_title("Post Event NBR")
    plt.show()

def fire_detector_nbr():
    """
    Detects fire-affected areas by comparing pre and post-fire NBR values.
    Creates and displays a visualization of the NBR difference.
    """
    # Open both NBR files using context managers
    with rasterio.open("NBR_PRE.tiff") as src_pre, rasterio.open("NBR_Post.tiff") as src_post:
        # Load NBR data into numpy arrays
        NBR_Pre = src_pre.read(1)
        NBR_Post = src_post.read(1)

        # Calculate NBR difference (positive values indicate burned areas)
        final_fire_NBR = NBR_Pre - NBR_Post

        # Create visualization
        f, ax = plt.subplots(1, 1, dpi=100, figsize=(18, 6))
        # Display difference map with value range [-1, 1]
        im = ax.imshow(final_fire_NBR, vmin=-1, vmax=1, cmap='Reds')

        # Add colorbar with label
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('NBR Value')

        ax.set_title("Fire Area - NBR Difference")
        plt.show()


def fire_area_nbr():
    """
    Detects and visualizes fire-affected areas using NBR (Normalized Burn Ratio) difference.
    Applies median filtering to reduce noise and excludes water bodies using NDWI.
    """
    # Set up noise reduction parameters
    kernel_size = 3  # Size of median filter window (3x3 pixels)
    iterations = 1   # Number of filtering passes

    # Load all required raster data using context managers
    with rasterio.open("NBR_PRE.tiff") as src_pre, \
         rasterio.open("NBR_Post.tiff") as src_post, \
         rasterio.open("NDWI.tiff") as src_ndwi:
        
        # Read raster data into numpy arrays
        NBR_Pre = src_pre.read(1)      # Pre-fire NBR
        NBR_Post = src_post.read(1)     # Post-fire NBR
        NDWI = src_ndwi.read(1)         # Water index

        # Calculate NBR difference (positive values indicate burned areas)
        final_fire_NBR = NBR_Pre - NBR_Post

    # Create fire detection mask:
    # - NBR difference > 0.25 indicates burned areas
    # - NDWI <= 0 excludes water bodies
    fire_condition = (final_fire_NBR > 0.25) & (NDWI <= 0)

    # Apply median filtering to reduce noise
    filtered_fire_condition = fire_condition
    for _ in range(iterations):
        filtered_fire_condition = median_filter(filtered_fire_condition, size=kernel_size)

    # Set up visualization parameters
    # Create binary colormap: black for no fire, red for fire
    cmap = matplotlib.colors.ListedColormap(["black", "firebrick"])
    values = ["Absence", "Presence"]
    colors = ["black", "firebrick"]

    # Create visualization
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(12, 6))

    # Display fire detection map
    im = show(
        filtered_fire_condition,
        vmin=0,                    # Minimum value (no fire)
        vmax=1,                    # Maximum value (fire)
        transform=src_pre.transform,# Maintain spatial reference
        ax=ax,
        cmap=cmap,
    )

    ax.set_title("Fire Area With NBR Mode")

    # Create and add legend
    patches = [
        mpatches.Patch(color=colors[i], label=f"Fire {values[i]}")
        for i in range(len(values))
    ]
    f.legend(handles=patches, bbox_to_anchor=(0.95, 0.2), loc=1)
    plt.tight_layout()
    plt.show()

def fire_area_ndvi():
    """
    Detects and visualizes fire-affected areas using NDVI (Normalized Difference Vegetation Index) difference.
    Applies median filtering to reduce noise and excludes water bodies using NDWI.
    """
    # Set up noise reduction parameters
    kernel_size = 3  # Size of median filter window (3x3 pixels)
    iterations = 1   # Number of filtering passes

    # Load all required raster data using context managers
    with rasterio.open("NDVI_PRE.tiff") as src_pre, \
         rasterio.open("Post_NDVI.tiff") as src_post, \
         rasterio.open("NDWI.tiff") as src_ndwi:
        
        # Read raster data into numpy arrays
        ndvi_pre = src_pre.read(1)   # Pre-fire NDVI
        ndvi_post = src_post.read(1)  # Post-fire NDVI
        ndwi = src_ndwi.read(1)       # Water index

        # Calculate NDVI difference (positive values indicate vegetation loss)
        final_fire_NDVI = ndvi_pre - ndvi_post

    # Apply median filtering to reduce noise in NDVI difference
    filtered_fire_condition = median_filter(final_fire_NDVI, size=kernel_size)

    # Set up visualization parameters
    # Create binary colormap: black for no fire, red for fire
    cmap = matplotlib.colors.ListedColormap(["black", "firebrick"])
    values = ["Absence", "Presence"]
    colors = ["black", "firebrick"]

    # Create fire detection mask:
    # - Filtered NDVI difference > 0.10 indicates potential burned areas
    # - ndwi <= 0 excludes water bodies
    fire_condition = (filtered_fire_condition > 0.10) & (ndwi <= 0)

    # Create visualization
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(12, 6))

    # Display fire detection map
    im = show(
        fire_condition,
        vmin=0,                    # Minimum value (no fire)
        vmax=1,                    # Maximum value (fire)
        transform=src_pre.transform,# Maintain spatial reference
        ax=ax,
        cmap=cmap,
    )

    ax.set_title("Fire Area with NDVI Mode (Excluding Water Areas)")

    # Create and add legend
    patches = [
        mpatches.Patch(color=colors[i], label=f"Fire {values[i]}")
        for i in range(len(values))
    ]
    f.legend(handles=patches, bbox_to_anchor=(0.95, 0.2), loc=1)
    plt.tight_layout()
    plt.show()


def severity_ndvi():
    """
    Analyzes and visualizes fire severity using NDVI (Normalized Difference Vegetation Index).
    Includes water body masking, severity classification, and area calculations.
    Generates a color-coded severity map and calculates affected areas by severity class.
    """
    # Set up noise reduction parameters
    kernel_size = 3  # Size of median filter window (3x3 pixels)
    iterations = 1   # Number of filtering passes

    def classify_burn_severity_ndvi(ndvi):
        """
        Classifies burn severity based on NDVI difference thresholds.
        
        Args:
            ndvi (float): NDVI difference value
            
        Returns:
            int: Severity class (0-4)
                4: High Severity (NDVI > 0.5)
                3: Moderate High Severity (0.3 < NDVI ≤ 0.5)
                2: Moderate Severity (0.2 < NDVI ≤ 0.3)
                1: Low Severity (0.17 < NDVI ≤ 0.2)
                0: Unburned (NDVI ≤ 0.17)
        """
        if ndvi > 0.5:
            return 4  # High Severity
        elif ndvi > 0.3:
            return 3  # Moderate High Severity
        elif ndvi > 0.2:
            return 2  # Moderate Severity
        elif ndvi > 0.17:
            return 1  # Low Severity
        else:
            return 0  # Unburned

    # Load NDVI data from pre and post-fire periods
    with rasterio.open("NDVI_PRE.tiff") as src_pre, rasterio.open("Post_NDVI.tiff") as src_post:
        ndvi_pre = src_pre.read(1)     # Pre-fire NDVI
        ndvi_post = src_post.read(1)    # Post-fire NDVI
        transform = src_pre.transform    # Save geospatial transform for plotting

    # Calculate NDVI difference (positive values indicate vegetation loss)
    fire_ndvi = ndvi_pre - ndvi_post

    # Load and apply water mask using NDWI
    with rasterio.open("NDWI.tiff") as src_ndwi:
        ndwi = src_ndwi.read(1)
    
    # Mask out water bodies by setting their values to NaN
    fire_ndvi[ndwi > 0] = np.nan

    # Apply median filtering to reduce noise
    for _ in range(iterations):
        fire_ndvi = median_filter(fire_ndvi, size=kernel_size)

    # Define visualization parameters
    cmap = mcolors.ListedColormap(["black", "yellow", "orange", "red", "darkred"])
    values = ["Unburned", "Low Severity", "Moderate Severity", 
              "Moderate High Severity", "High Severity"]

    # Create severity classification map
    severity_map_mosaic = np.zeros_like(fire_ndvi, dtype=int)
    for i in range(fire_ndvi.shape[0]):
        for j in range(fire_ndvi.shape[1]):
            # Only classify non-water areas (not NaN)
            if not np.isnan(fire_ndvi[i, j]):
                severity_map_mosaic[i, j] = classify_burn_severity_ndvi(fire_ndvi[i, j])

    # Create visualization
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(12, 6))
    show(
        severity_map_mosaic,
        vmin=0,                # Minimum value (unburned)
        vmax=4,                # Maximum value (high severity)
        transform=transform,    # Maintain spatial reference
        ax=ax,
        cmap=cmap
    )
    ax.set_title("Fire Severity (NDVI) - Excluding Water")

    # Create and add legend
    patches = [
        mpatches.Patch(color=cmap(i), label=f"{values[i]}")
        for i in range(len(values))
    ]
    f.legend(handles=patches, bbox_to_anchor=(0.95, 0.2), loc=1)

    # Calculate area for each severity class
    severity_classes = [0, 1, 2, 3, 4]
    area_per_class_km2 = {}
    
    # Define pixel size and convert to km²
    pixel_area_m2 = 10 * 10  # 10m x 10m Sentinel-2 pixel
    pixel_area_km2 = pixel_area_m2 / 1_000_000  # Convert to km²

    # Calculate area for each severity class
    for severity in severity_classes:
        area_per_class_km2[severity] = np.count_nonzero(severity_map_mosaic == severity) * pixel_area_km2

    # Print area statistics
    print("\nArea (km²) in Fire NDVI mode (excluding water):")
    for severity, area in area_per_class_km2.items():
        print(f"{values[severity]}: {area:.2f} km²")

    # Calculate and print total burned area (excluding unburned class)
    total_fire_area = sum(area for severity, area in area_per_class_km2.items() if severity > 0)
    print(f"\nTotal Fire Area (km²): {total_fire_area:.2f} km²")

    # Display the final plot
    plt.tight_layout()
    plt.show()


def severity_nbr():
    """
    Analyzes and visualizes fire severity using NBR (Normalized Burn Ratio).
    Includes water body masking, severity classification, and area calculations.
    Generates a color-coded severity map and calculates affected areas by severity class.
    """
    # Set up noise reduction parameters
    kernel_size = 3  # Size of median filter window (3x3 pixels)
    iterations = 1   # Number of filtering passes

    def classify_burn_severity_nbr(nbr_diff):
        """
        Classifies burn severity based on NBR difference thresholds.
        
        Args:
            nbr_diff (float): NBR difference value
            
        Returns:
            int: Severity class (0-4)
                4: High Severity (NBR > 0.70)
                3: Moderate High Severity (0.50 < NBR ≤ 0.70)
                2: Moderate Severity (0.30 < NBR ≤ 0.50)
                1: Low Severity (0.20 < NBR ≤ 0.30)
                0: Unburned (NBR ≤ 0.20)
        """
        if nbr_diff > 0.70:
            return 4  # High Severity
        elif nbr_diff > 0.50:
            return 3  # Moderate High Severity
        elif nbr_diff > 0.30:
            return 2  # Moderate Severity
        elif nbr_diff > 0.20:
            return 1  # Low Severity
        else:
            return 0  # Unburned

    # Load NBR data from pre and post-fire periods
    with rasterio.open("NBR_PRE.tiff") as src_pre, rasterio.open("NBR_Post.tiff") as src_post:
        NBR_Pre = src_pre.read(1)      # Pre-fire NBR
        NBR_Post = src_post.read(1)     # Post-fire NBR
        transform = src_pre.transform    # Save geospatial transform for plotting

    # Calculate NBR difference (positive values indicate burned areas)
    final_fire_NBR = NBR_Pre - NBR_Post

    # Load and apply water mask using NDWI
    with rasterio.open("NDWI.tiff") as src_ndwi:
        ndwi = src_ndwi.read(1)
    
    # Mask out water bodies by setting their values to NaN
    final_fire_NBR[ndwi > 0] = np.nan

    # Apply median filtering to reduce noise
    for _ in range(iterations):
        final_fire_NBR = median_filter(final_fire_NBR, size=kernel_size)

    # Define visualization parameters
    # Color scheme: black (unburned) to dark red (high severity)
    cmap = mcolors.ListedColormap(["black", "yellow", "orange", "red", "darkred"])
    values = ["Unburned", "Low Severity", "Moderate Severity", 
              "Moderate High Severity", "High Severity"]

    # Create severity classification map
    severity_map_nbr = np.zeros_like(final_fire_NBR, dtype=int)
    for i in range(final_fire_NBR.shape[0]):
        for j in range(final_fire_NBR.shape[1]):
            # Only classify non-water areas (not NaN)
            if not np.isnan(final_fire_NBR[i, j]):
                severity_map_nbr[i, j] = classify_burn_severity_nbr(final_fire_NBR[i, j])

    # Create visualization
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(12, 6))
    show(
        severity_map_nbr,
        vmin=0,                # Minimum value (unburned)
        vmax=4,                # Maximum value (high severity)
        transform=transform,    # Maintain spatial reference
        ax=ax,
        cmap=cmap
    )
    ax.set_title("Fire Severity Based on NBR (Excluding Water)")

    # Create and add legend
    patches = [
        mpatches.Patch(color=cmap(i), label=f"{values[i]}")
        for i in range(len(values))
    ]
    f.legend(handles=patches, bbox_to_anchor=(0.95, 0.2), loc=1)

    # Calculate area for each severity class
    severity_classes = [0, 1, 2, 3, 4]
    area_per_class_nbr = {}
    
    # Define pixel size and convert to km²
    pixel_area_m2 = 10 * 10  # 10m x 10m Sentinel-2 pixel
    pixel_area_km2 = pixel_area_m2 / 1_000_000  # Convert to km²

    # Calculate area for each severity class
    for severity in severity_classes:
        area_per_class_nbr[severity] = np.count_nonzero(severity_map_nbr == severity) * pixel_area_km2

    # Print area statistics
    print("\nArea (km²) in Fire mode (NBR, Excluding Water):")
    for severity, area in area_per_class_nbr.items():
        print(f"{values[severity]}: {area:.2f} km²")

    # Calculate and print total burned area (excluding unburned class)
    total_fire_area = sum(area for severity, area in area_per_class_nbr.items() if severity > 0)
    print(f"\nTotal Fire Area (km²): {total_fire_area:.2f} km²")

    # Display the final plot
    plt.tight_layout()
    plt.show()


def fire_severity_multiclass():
    # Open the pre-fire NDWI raster file to mask water areas later
    with rasterio.open("NDWI.tiff") as src_pre:
        # Read the NDWI data into a numpy array
        ndwi = src_pre.read(1)
    
    # Open the pre-fire and post-fire NBR raster files
    with rasterio.open("NBR_PRE.tiff") as src_pre, rasterio.open("NBR_Post.tiff") as src_post:
        # Read the NBR data for pre-fire and post-fire conditions
        NBR_Pre = src_pre.read(1)
        NBR_Post = src_post.read(1)
        # Save the transform for plotting
        transform = src_pre.transform

    # Define parameters for the median filter to smooth the data
    kernel_size = 3  # Size of the filter kernel (3x3 neighborhood)
    iterations = 1  # Number of times to apply the filter

    # Calculate the difference in NBR (Pre-fire - Post-fire)
    final_fire_NBR = NBR_Pre - NBR_Post

    # Mask water areas by setting pixels where NDWI > 0 to NaN
    final_fire_NBR[ndwi > 0] = np.nan

    # Apply a median filter to smooth the NBR difference
    for _ in range(iterations):
        final_fire_NBR = median_filter(final_fire_NBR, size=kernel_size)

    # Open the pre-fire and post-fire NDVI raster files
    with rasterio.open("NDVI_PRE.tiff") as src_pre, rasterio.open("Post_NDVI.tiff") as src_post:
        # Read the NDVI data for pre-fire and post-fire conditions
        ndvi_pre = src_pre.read(1)
        ndvi_post = src_post.read(1)
        # Save the transform for plotting
        transform = src_pre.transform

    # Calculate the difference in NDVI (Pre-fire - Post-fire)
    fire_ndvi = ndvi_pre - ndvi_post

    # Mask water areas by setting pixels where NDWI > 0 to NaN
    fire_ndvi[ndwi > 0] = np.nan

    # Apply a median filter to smooth the NDVI difference
    for _ in range(iterations):
        final_fire_ndvi = median_filter(fire_ndvi, size=kernel_size)

    # Define a function to classify fire severity based on NBR and NDVI values
    def classify_fire_area_multi(NBR_Post, ndvi_post, final_fire_ndvi, final_fire_nbr, ndwi):
        if ndwi > 0.0:  # Exclude water areas
            return 0  # Unaffected Area (Water)
        elif final_fire_nbr > 0.7 or (final_fire_ndvi > 0.50 and NBR_Post < 0.1 and ndvi_post < 0.0):
            return 4  # Very High Severity Fire
        elif final_fire_nbr > 0.5 or (final_fire_ndvi > 0.40 and NBR_Post < 0.2 and ndvi_post < 0.05):
            return 3  # High Severity Fire
        elif final_fire_nbr > 0.3 or (final_fire_ndvi > 0.30 and NBR_Post < 0.25 and ndvi_post < 0.10):
            return 2  # Moderate Severity Fire
        elif final_fire_nbr > 0.25 or (final_fire_ndvi > 0.20 and NBR_Post < 0.3 and ndvi_post < 0.15):
            return 1  # Low Severity Fire
        else:
            return 0  # Unaffected Area

    # Create a colormap for visualizing fire severity classes
    cmap = mcolors.ListedColormap(["black", "yellow", "orange", "red", "darkred"])
    labels = ["Unburned", "Low Severity", "Moderate Severity", "Moderate High Severity", "High Severity"]

    # Initialize an array to store the classified fire severity
    fire_area_map_multi = np.zeros_like(ndwi, dtype=int)
    rows, cols = fire_area_map_multi.shape

    # Iterate over each pixel to classify fire severity
    for i in range(rows):
        for j in range(cols):
            fire_area_map_multi[i, j] = classify_fire_area_multi(
                NBR_Post[i, j], ndvi_post[i, j], final_fire_ndvi[i, j], final_fire_NBR[i, j], ndwi[i, j]
            )

    # Plot the classified fire severity map
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(12, 6))
    show(
        fire_area_map_multi,
        transform=transform,
        cmap=cmap,
        ax=ax,
    )
    ax.set_title("Detected Fire Area (Multi-Class)")

    # Add a legend for the fire severity classes
    patches = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(len(labels))]
    f.legend(handles=patches, bbox_to_anchor=(0.95, 0.2), loc=1)

    # Calculate the area (in km²) for each severity classs
    class_areas = {}
    pixel_area_km2 = (10 * 10) / 1_000_000  # Assuming 10 m resolution
    for class_value in range(5):  # Classes 0 to 4
        class_pixel_count = np.count_nonzero(fire_area_map_multi == class_value)
        class_areas[class_value] = class_pixel_count * pixel_area_km2

    # Calculate the total fire-affected area (sum of areas for severity classes 1 to 4)
    total_fire_area = sum(area for severity, area in class_areas.items() if severity > 0)

    # Print the area for each severity class
    print("\nFire-Affected Area by Severity (km²):")
    for class_value, area in class_areas.items():
        print(f"{labels[class_value]}: {area:.2f} km²")

    # Print the total fire area
    print(f"\nTotal Fire Area (km²): {total_fire_area:.2f} km²")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def severity_kmeans():
    # Open the NDWI raster file (used for water masking)
    with rasterio.open("NDWI.tiff") as src_ndwi:
        ndwi = src_ndwi.read(1)  # Read NDWI raster data as a numpy array

    # Open the post-fire NDVI raster file
    with rasterio.open("Post_NDVI.tiff") as src_post:
        ndvi_post = src_post.read(1)  # Read post-fire NDVI raster data

    # Open the post-fire NBR raster file
    with rasterio.open("NBR_Post.tiff") as src_post:
        nbr_post = src_post.read(1)  # Read post-fire NBR raster data

    # Define parameters for the median filter to smooth data
    kernel_size = 3  # Size of the kernel (e.g., 3x3 neighborhood)
    iterations = 1  # Number of times the filter is applied

    # Open the pre-fire NBR raster file
    with rasterio.open("NBR_PRE.tiff") as src_pre:
        nbr_pre = src_pre.read(1)  # Read pre-fire NBR raster data
        transform = src_pre.transform  # Save the geospatial transform for plotting

    # Calculate the difference in NBR (Pre-fire - Post-fire)
    final_fire_NBR = nbr_pre - nbr_post

    # Mask water areas where NDWI > 0 by setting them to NaN
    final_fire_NBR[ndwi > 0] = np.nan

    # Apply a median filter to smooth the NBR difference
    for _ in range(iterations):
        final_fire_NBR = median_filter(final_fire_NBR, size=kernel_size)

    # Open the pre-fire and post-fire NDVI raster files
    with rasterio.open("NDVI_PRE.tiff") as src_pre, rasterio.open("Post_NDVI.tiff") as src_post:
        ndvi_pre = src_pre.read(1)  # Read pre-fire NDVI raster data
        ndvi_post = src_post.read(1)  # Read post-fire NDVI raster data
        transform = src_pre.transform  # Save the geospatial transform for plotting

    # Calculate the difference in NDVI (Pre-fire - Post-fire)
    fire_ndvi = ndvi_pre - ndvi_post

    # Mask water areas where NDWI > 0 by setting them to NaN
    fire_ndvi[ndwi > 0] = np.nan

    # Apply a median filter to smooth the NDVI difference
    for _ in range(iterations):
        final_fire_NDVI = median_filter(fire_ndvi, size=kernel_size)

    # Rename for clarity
    final_fire_ndvi = final_fire_NDVI  # Smoothed NDVI difference
    final_fire_nbr = final_fire_NBR   # Smoothed NBR difference

    # Stack features (NDVI, NBR, and post-fire data) for K-means clustering
    features = np.stack([final_fire_ndvi.flatten(), final_fire_nbr.flatten(), nbr_post.flatten(), ndvi_post.flatten()], axis=-1)

    # Impute NaN values with the mean for each feature
    imputer = SimpleImputer(strategy='mean')  # Create an imputer object
    features_imputed = imputer.fit_transform(features)  # Impute missing values

    # Normalize the features for K-means clustering (standardize scale)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Perform K-means clustering
    n_clusters = 4  # Number of clusters (adjust as needed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(features_scaled)  # Cluster assignment

    # Reshape labels into the shape of the original raster
    fire_area_map_multi = kmeans_labels.reshape(ndwi.shape)

    # Mask water areas (NDWI > 0) by setting them to -1 (No Data)
    fire_area_map_multi[ndwi > 0] = -1

    # Define a colormap dynamically based on the number of clusters
    colors = ["black", "red", "orange", "yellow", "green", "blue", "purple"][:n_clusters]
    cmap = mcolors.ListedColormap(colors)
    labels = [f"Cluster {i + 1}" for i in range(n_clusters)]

    # Plot the fire severity classification based on K-means clustering
    f, ax = plt.subplots(1, 1, dpi=100, figsize=(12, 6))
    show(
        fire_area_map_multi,
        transform=src_ndwi.transform,
        cmap=cmap,
        ax=ax,
    )
    ax.set_title(f"Fire Area Classification Using K-Means Clustering ({n_clusters} Clusters)")

    # Add a legend for the clusters
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n_clusters)]
    f.legend(handles=patches, bbox_to_anchor=(0.95, 0.2), loc=1)

    # Calculate area for each cluster
    pixel_area_m2 = 10 * 10  # Assuming 10 m resolution
    pixel_area_km2 = pixel_area_m2 / 1_000_000  # Convert to square kilometers

    cluster_areas = {}
    for cluster_id in range(n_clusters):
        cluster_area = np.count_nonzero(fire_area_map_multi == cluster_id) * pixel_area_km2
        cluster_areas[cluster_id] = cluster_area
        print(f"Cluster {cluster_id + 1} Area (km²): {cluster_area:.2f} km²")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


