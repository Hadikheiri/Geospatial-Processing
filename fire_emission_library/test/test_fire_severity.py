### tests/test_fire_severity.py
import unittest
import numpy as np
from fire_emission_analysis.fire_severity import calculate_fire_severity

class TestFireSeverity(unittest.TestCase):
    def test_calculate_fire_severity(self):
        ndvi_diff = np.array([0.1, 0.4, 0.7])
        thresholds = {1: 0.2, 2: 0.5}
        severity = calculate_fire_severity(ndvi_diff, thresholds)
        self.assertEqual(severity.tolist(), [0, 1, 2])
