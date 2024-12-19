### fire_emission_analysis/land_cover.py
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def classify_land_cover(features, labels, region_features):
    """Perform SVM classification for land cover."""
    scaler = StandardScaler()
    svm = SVC()
    scaled_features = scaler.fit_transform(features)
    svm.fit(scaled_features, labels)
    scaled_region = scaler.transform(region_features)
    classified_map = svm.predict(scaled_region)
    return classified_map
