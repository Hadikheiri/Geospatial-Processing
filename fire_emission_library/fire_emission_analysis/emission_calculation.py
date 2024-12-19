### fire_emission_analysis/emission_calculation.py

def calculate_emissions(classified_area, emission_factors):
    """Estimate CO2 emissions based on land cover types and emission factors."""
    emissions = 0
    for land_type, area in classified_area.items():
        emissions += area * emission_factors.get(land_type, 0)
    return emissions