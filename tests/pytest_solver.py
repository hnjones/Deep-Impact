import pandas as pd
import pytest
from deepimpact import Planet  # Replace 'solver' with the actual module name
import numpy as np

data = np.load('tests/scenario.npz') 

# Length consistency test
lengths = [len(data[key]) for key in data.files]
assert all(length == lengths[0] for length in lengths), "All arrays should have the same length"

# Range validity tests
assert all(data['mass'] >= 0), "Mass values should be non-negative"
assert all(data['radius'] >= 0), "Radius values should be non-negative"
assert all(data['altitude'] >= 0), "Altitude values should be non-negative"
assert all(0 <= angle <= 90 for angle in data['angle']), "Angles should be between 0 and 90 degrees"

# Physical plausibility tests
# Example: Checking if mass decreases monotonically (if that's expected)
assert all(m2 <= m1 for m1, m2 in zip(data['mass'], data['mass'][1:])), "Mass should decrease monotonically"

# Check that all arrays have the same length
lengths = [len(data[key]) for key in data.files]
assert all(length == lengths[0] for length in lengths), "Arrays should have the same length"



  # Check for non-negative values in certain arrays (e.g., mass, radius, altitude)
assert all(data['mass'] >= 0), "Mass values should be non-negative"
assert all(data['radius'] >= 0), "Radius values should be non-negative"
assert all(data['altitude'] >= 0), "Altitude values should be non-negative"

# Check that angles are within a valid range (0 to 90 degrees for example)
assert all(0 <= angle <= 90 for angle in data['angle']), "Angles should be between 0 and 90 degrees"


def test_analyse_outcome():
    planet = Planet()

    # Creating a dummy DataFrame with sample data for testing
    sample_data = {
        'velocity': [3000, 1500],  # Sample velocities
        'mass': [1000, 500],  # Sample masses
        'angle': [45, 30],  # Sample angles
        'altitude': [10000, -5],  # Altitudes (second one simulates impact)
        'distance': [20000, 25000],  # Sample distances
        'radius': [10, 5],  # Sample radii
        'dedz': [1e5, 2e5]  # Sample energy deposition rates
    }
    test_df = pd.DataFrame(sample_data)

    # Calling the analyse_outcome method
    outcome = planet.analyse_outcome(test_df)

    # Asserting the outcome dictionary contains expected keys
    expected_keys = ['outcome', 'burst_peak_dedz', 'burst_altitude', 'burst_distance', 'burst_energy']
    assert all(key in outcome for key in expected_keys)

    # Test for Airburst outcome
    assert outcome['outcome'] == 'Airburst'
    assert outcome['burst_peak_dedz'] == 2700  # Asserting the peak energy deposition rate
    assert outcome['burst_altitude'] == test_df.loc[test_df['dedz'].idxmax(), 'altitude']

    # Adjusting data to test for Cratering outcome
    test_df.loc[test_df.index[-1], 'altitude'] = 0
    cratering_outcome = planet.analyse_outcome(test_df)
    assert cratering_outcome['outcome'] == 'Cratering'


