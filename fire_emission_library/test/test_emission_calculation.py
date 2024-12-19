### tests/test_emission_calculation.py
import unittest
from fire_emission_analysis.emission_calculation import calculate_emissions

class TestEmissionCalculation(unittest.TestCase):
    def test_calculate_emissions(self):
        classified_area = {"forest": 10, "grassland": 5}
        emission_factors = {"forest": 1.2, "grassland": 0.8}
        emissions = calculate_emissions(classified_area, emission_factors)
        self.assertAlmostEqual(emissions, 16.0)

if __name__ == "__main__":
    unittest.main()