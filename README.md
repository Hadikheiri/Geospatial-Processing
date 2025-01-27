# Atash: Fire Detection and Severity Classification using Remote Sensing Data

Atash is an open-source Python library designed for fire detection and burn severity classification using remote sensing data. It integrates with OpenEO to process Sentinel-2 imagery and generates detailed NDVI, NBR, and NDWI maps for analysis.

![image](https://github.com/user-attachments/assets/782b5d71-eb4d-45cb-bd68-5819d28effe0)


## Table of Contents

1. [Authentication Details](#authentication-details)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Input/Output Specifications](#inputoutput-specifications)
8. [Function Documentation](#function-documentation)
9. [Error Handling](#error-handling)
10. [Performance Considerations](#performance-considerations)
11. [Contributors and Support](#contributors-and-support)
12. [License](#license)
13. [Data Privacy Note](#data-privacy-note)

---

## 1. Authentication Details

To authenticate with OpenEO, you need OIDC credentials.

1. Register at [Copernicus Data Space](https://dataspace.copernicus.eu) to obtain login credentials.
2. Create a connection with Copernicus API:
   ```
   openeo:
     connection = connect_to_openeo()
   ```

---

## 2. Features

- **OpenEO Integration**: Easily connect and authenticate using OIDC.
- **NDVI, NBR, and NDWI Calculations**: Compute vegetation health, burn severity, and water indices.
- **Fire Detection**: Detect fire-affected areas using NDVI and NBR differences.
- **Severity Classification**: Classify burn severity using thresholds and KMeans clustering.
  - **KMeans Clustering**: Groups pixels into clusters based on NDVI and NBR values to classify fire severity levels.
- **Interactive ROI Selection**: Use an interactive map to define spatial extents.
- **Water Masking**: Improve accuracy by excluding water bodies using NDWI.

---

## 3. File Structure

```
atash/
├── __init__.py          # Package initialization file
├── atash.py             # Library source code for fire detection and classification
License
README.md                # Project documentation
Test.ipynb               # Jupyter notebook for testing and demonstration
setup.py                 # Script for library installation

```

---

## 4. Installation

### Prerequisites

- Python 3.6+
- Pip and Git


- Install with pip
   ```python
   !pip install atash
   ```
   

---



## 5. Usage

### Complete Workflow Example

```python
from atash import connect_to_openeo, load_map, get_start_and_end_dates, load_pre_ndvi, load_post_ndvi, fire_detector_ndvi, severity_ndvi

# Step 1: Connect to OpenEO
connection = connect_to_openeo()

# Step 2: Define area of interest
map_interface = load_map()

# Step 3: Set temporal extent
start_date, end_date = get_start_and_end_dates()

# Step 4: Process pre/post-event data
load_pre_ndvi(connection, extent, start_date, end_date)
load_post_ndvi(connection, extent, start_date, end_date)

# Step 5: Analyze fire impact
fire_detector_ndvi()
severity_ndvi()
```

---

## 6. Input/Output Specifications

### Input Data:

- **Format**: `.tiff` files for raster data from openeo.
- **Required Bands**:
  - B03 (Green), B04 (Red), B08 (NIR), B12 (SWIR).
- **Coordinate Reference System (CRS)**: Input rasters must have the same CRS (e.g., `EPSG:4326`).
- **Raster Resolution**: Assumed to be 10m x 10m.

### File Naming Conventions:

- Ensure consistent file naming:
  - `NDVI_PRE.tiff` for pre-event NDVI.
  - `NBR_PRE.tiff` for pre-event NBR.
  - `NDWI.tiff` for water mask rasters.
  - `NDVI_Post.tiff` for post-event NDVI.
  - `NBR_Post.tiff` for post-event NBR.
### Output Files and Results:

- ``: Pre-event NDVI raster.
- ``: Post-event NDVI raster.
- ``: Pre-event NBR raster.
- ``: Post-event NBR raster.
- ``: Water mask raster.
- ``: Wildfire Severity by NDVI and surface area of them.
- ``: Wildfire Severity by NBR and surface area of them.
- ``: Wildfire Severity by NDVI and NBR and surface area of them.
- ``: K-Means Clustering of wildfire.
---

## 8. Function Documentation

### 

| Function                    | Purpose                                   | Parameters                                 |
| --------------------------- | ----------------------------------------- | ------------------------------------------ |
| `connect_to_openeo()`       | Establishes OpenEO connection             | -                                          |
| `load_map()`                | Displays an interactive map for ROI       | -                                          |
| `get_start_and_end_dates()` | Captures start and end dates              | -                                          |
| `load_pre_ndvi()`           | Loads pre-event NDVI data                 | `connection, extent, start_date, end_date` |
| `load_post_ndvi()`          | Loads post-event NDVI data                | `connection, extent, start_date, end_date` |
| `load_pre_nbr()`            | Loads pre-event NBR data                  | `connection, extent, start_date, end_date` |
| `load_post_nbr()`           | Loads post-event NBR data                 | `connection, extent, start_date, end_date` |
| `fire_detector_nbr()`       | Detects fire-affected pixels via NBR      | -                                          |
| `fire_area_nbr()`           | Detects fire-affected areas via NBR       | -                                          |
| `fire_area_ndvi()`          | Detects fire-affected areas via NDVI      | -                                          |
| `severity_ndvi()`           | Classifies fire severity (NDVI)           | -                                          |
| `severity_nbr()`            | Uses KMeans clustering for classification | -                                          |
| `severity_kmeans()`         | Uses KMeans clustering for classification | -                                          |


---

## 9. Error Handling

### Common Issues and Solutions

1. **OpenEO Connection Error**:

   - **Message**: "Failed to connect to OpenEO: [specific error]"
   - **Solution**: Check the OpenEO URL, credentials, and internet connection.

2. **FileNotFoundError**:

   - **Message**: "File not found: `NDVI_PRE.tiff`"
   - **Solution**: Ensure the required `.tiff` files are present in the folder.

3. **MemoryError**:

   - **Message**: "Unable to allocate array"
   - **Solution**: Reduce raster size or increase system memory.

4. **Data Format Issue**:

   - **Message**: "Invalid raster dimensions"
   - **Solution**: Ensure input `.tiff` files have matching resolutions and CRS.

5. **Median Filter Missing Dependency**:

   - **Message**: "ModuleNotFoundError: No module named 'scipy'"
   - **Solution**: Install `scipy` using `pip install scipy`.

---

## 10. Performance Considerations

- **Processing Time**:
  - Small ROI (<500 km²): \~15 minutes.
  - Large ROI (>1000 km²): \~30 minutes.
- **System Requirements**:
  - 8GB RAM (minimum).
  - Multi-core CPU recommended.

---

## 11. Contributors and Support
We extend our gratitude to the contributors of the "Atash" project. You can explore their profiles for more projects and collaborations:

- **Bahman Amirsardary** – [bahman.amirsardary@mail.polimi.it](mailto:bahman.amirsardary@mail.polimi.it).
- **Hadi Kheiri Gharajeh** – [hadi.kheiri@mail.polimi.it](mailto:hadi.kheiri@mail.polimi.it).
- **Milad Ramezani Ziarani** – [milad.ramezani@mail.polimi.it](mailto:milad.ramezani@mail.polimi.it).

For support or contributions, feel free to contact us.



---

## 12. License

Atash is licensed under the [MIT License](LICENSE). This means:

- **Permission is Granted to:**
- Use the software for personal, academic, or commercial purposes.
- Modify and distribute the code under the same license.

- **Limitations:**
- The software is provided "as is," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.

For the full license text, see the `LICENSE` file in the repository.

---

## 13. Data Privacy Note

Atash does not store or log user data. Any authentication or API usage is handled securely via OpenEO's OIDC protocol. Ensure you review OpenEO's privacy policy for their data handling practices.

---

Contributions are welcome! Feel free to open issues or submit pull requests.
