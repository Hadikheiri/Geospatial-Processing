### fire_emission_analysis/pollution_analysis.py
import numpy as np

def detect_smoke_extent(smoke_index, threshold):
    """Detect the area covered by smoke."""
    smoke_area = np.sum(smoke_index > threshold)
    return smoke_area

def generate_pollution_heatmap(smoke_index):
    """Generate a heatmap of pollution intensity."""
    return smoke_index / np.max(smoke_index)
