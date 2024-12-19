def calculate_fire_severity(band_data, thresholds):
    """
    Classify fire severity based on band data and thresholds.
    Example thresholds: {'low': 0.2, 'medium': 0.5, 'high': 0.8}
    """
    severity_map = np.zeros_like(band_data)
    for severity, threshold in thresholds.items():
        severity_map[band_data > threshold] = severity
    return severity_map
