# Fire Detection and Severity Classification using Remote Sensing Data

## Overview
This project aims to classify and detect fire-affected areas and assess the severity of fires using remote sensing data, including NDVI, NBR, and NDWI rasters. The project leverages machine learning techniques, particularly K-means clustering, to differentiate between burned and unburned areas, and classify the severity of fires into multiple categories based on post-fire satellite imagery.

## Contributors
- **Bahman Amirsardary**
- **Hadi Kheiri**
- **Milad Ramezani Ziarani**

## Project Objective
The main objective of this project is to:
- Use satellite data (NDVI, NBR, NDWI) to detect fire-affected areas.
- Classify fire severity into different categories using K-means clustering and threshold-based methods.
- Calculate and visualize the area affected by fires and their respective severity levels.

## Technologies Used
- **Python**: Programming language used for data analysis and processing.
- **NumPy**: For numerical operations and handling raster data.
- **Matplotlib**: For plotting maps and visualizing results.
- **Rasterio**: For reading and processing raster data files.
- **Scikit-learn**: For machine learning algorithms (K-means clustering and feature scaling).
- **SimpleImputer**: For handling missing data in the input features.

## Dataset
This project uses the following raster datasets:
- **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation health.
- **NBR (Normalized Burn Ratio)**: A fire detection index used to assess burn severity.
- **NDWI (Normalized Difference Water Index)**: Used to mask out water areas during fire classification.

These datasets are typically provided by satellite imagery such as MODIS or Landsat and processed into raster files (`.tiff` format).

## Project Workflow
1. **Data Preprocessing**:
   - Read the NDVI, NBR, and NDWI raster files.
   - Apply water masking based on NDWI values to exclude water areas from analysis.

2. **Feature Stacking**:
   - Stack the NDVI, NBR, and other relevant features into a single array for clustering.

3. **Clustering with K-means**:
   - Normalize the features.
   - Apply K-means clustering to classify pixels into different severity classes.

4. **Severity Classification**:
   - Classify the fire severity into multiple classes using thresholds or clustering results.
   - Generate a fire severity map and calculate the area affected by each severity class.

5. **Area Calculation**:
   - Calculate and print the area of burned and unburned regions.
   - Visualize the classified fire area map.


## Results
The script generates a classification map showing different severity levels of fire-affected areas. It also calculates the area (in square kilometers) for each fire severity class and outputs the results.

