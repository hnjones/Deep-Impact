from collections import OrderedDict
import pandas as pd
import numpy as np
import os
import sys
import pytest
from deepimpact.solver import Planet 

# BASE_PATH = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(BASE_PATH))
from pytest import fixture, mark
# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly
@fixture(scope='module')
def deepimpact():
    import deepimpact
    return deepimpact


@fixture(scope='module')
def planet(deepimpact):
    return deepimpact.Planet()


@fixture(scope='module')
def loc(deepimpact):
    return deepimpact.GeospatialLocator()


@fixture(scope='module')
def result(planet):
    input = {'radius': 1.,
             'velocity': 2.0e4,
             'density': 3000.,
             'strength': 1e5,
             'angle': 30.0,
             'init_altitude': 100000.0,
             }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope='module')
def outcome(planet, result):
    outcome = planet.analyse_outcome(result=result)
    return outcome


def test_import(deepimpact):
    assert deepimpact


def test_planet_signature(deepimpact):
    inputs = OrderedDict(atmos_func='exponential',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)


    # call by keyword
    _ = deepimpact.Planet(**inputs)


    # call by position
    _ = deepimpact.Planet(*inputs.values())


def test_attributes(planet):
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)


def test_atmos_filename(planet):

    assert os.path.isfile(planet.atmos_filename)


def test_solve_atmospheric_entry(result):

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    # Verify that the result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    # Check that numeric columns contain appropriate data types
    for column in ['velocity', 'mass', 'angle', 'altitude', 'distance', 'radius']:
        assert pd.api.types.is_numeric_dtype(result[column]), f"Column '{column}' should have a numeric data type"

    # Check for non-negative values where applicable (e.g., mass, radius)
    for column in ['mass', 'radius']:
        # assert (result[column] >= 0).all(), f"Column '{column}' should contain only non-negative values"

    # Check that the time column is sorted, if it should be
        assert (result['time'].sort_values() == result['time']).all(), "Time column should be sorted"


def load_expected_data():
    # Load the expected data from scenario.npz
    scenario_file = os.sep.join((os.path.dirname(__file__), 'scenario.npz'))
    data = np.load(scenario_file) 
    df = pd.DataFrame()
    for key in data.files:
        df[key] =  data[key]
    return df


def test_scenario(planet):
    inputs = {'radius': 35.,
              'angle': 45.,
              'strength': 1e7,
              'density': 3000.,
              'velocity': 19e3,
              'init_altitude': 100000.0}

    # Run the simulation
    result = planet.solve_atmospheric_entry(**inputs)
    print(result.velocity.head(40))


    assert type(result) is pd.DataFrame
    for key in ('velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'):
     assert key in result.columns


    # Load the expected scenario data
    expected_data = load_expected_data()

    # Compare the result with expected data
    # This assumes both result and expected_data are DataFrames with the same structure
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    num_rows_to_compare = 20  #
    for column in expected_data.columns:
        if column not in inputs:  # Skip the columns that are part of the inputs
            assert column in result.columns, f"Column '{column}' not found in result DataFrame"
            assert np.allclose(result[column].iloc[:num_rows_to_compare], 
                               expected_data[column].iloc[:num_rows_to_compare], 
                               rtol=1), f"{column} data does not match expected results in the first {num_rows_to_compare} rows"


    # Get the last row of expected_data
    last_row_expected = expected_data.iloc[-1]
    result_at_9_75 = result.iloc[39]

    # Check if the closest row in result at time = 9.75 matches the values of the last row of expected_data
    for column in expected_data.columns:
        if column not in inputs: 
            assert np.isclose(result_at_9_75[column], last_row_expected[column], atol=1), f"{column} at time 9.75 does not match expected value from scenario.npz"



# Define the Planet instance with specified parameters
simple_planet = Planet(
    Cd=1.0,
    Ch=0,
    Q=np.inf,
    Cl=0,
    alpha=0,
    Rp=np.inf,
    g=0,
    H=8000.0,
    rho0=1.2,
)

# Define an analytical equation for comparison in tests
def analytical_equation(altitude):
    Cd = 1.0
    Ch = 0
    Cl = 0
    v0 = 19000.0
    z0 = 100000.0  # Initial height in meters
    m = 5.387831e08  # Mass in kg
    theta = np.radians(45)  # Convert angle to radians
    H = 8000.0  # Atmospheric scale height in meters
    r = 35.0
    g = 0
    rho0 = 1.2

    K = (Cd * rho0 * np.pi * r**2) * H / (2 * m * np.sin(theta))
    B = v0 / (np.exp(np.exp(-z0 / H) * K))

    return B * np.exp(np.exp(-altitude / H) * K)

# Decorate this function with @pytest.fixture to make it a fixture
@pytest.fixture
def approx_results_fixture():
    inputs = {
        "radius": 35.0,
        "angle": 45.0,
        "strength": 1e7,
        "density": 3000.0,
        "velocity": 19e3,
    }

    return simple_planet.solve_atmospheric_entry(**inputs)

# Test function using the fixture
def test_no_gravity_flat_scenario(approx_results_fixture):
    for index, row in approx_results_fixture.iterrows():
        calculated_velocity = row["velocity"]
        approx_velocity = analytical_equation(row["altitude"])
        assert np.isclose(
            calculated_velocity, approx_velocity, rtol=0.1
        ), f"Row {index}: Calculated Velocity is not close to Approx Velocity"




@pytest.fixture
def earth():
    # Create an instance of the Planet class with tabular atmospheric function
    return Planet(atmos_func='tabular')

def test_known_data_points(earth):
    # Test at known data points from the CSV
    assert np.isclose(earth.rhoa(0), 1.225, atol=1e-3)
    assert np.isclose(earth.rhoa(2000), 1.00649, atol=1e-3)
    assert np.isclose(earth.rhoa(10000), 0.4127, atol=1e-3)
    # Test higher altitude data point from the CSV
    assert np.isclose(earth.rhoa(86000), 5.6411400000000003e-06, atol=1e-5)

def test_interpolation(earth):
    # Test the interpolation at points not directly in your dataset. Testing against standard atmosphere values
    # https://www.eoas.ubc.ca/courses/atsc113/flying/met_concepts/02-met_concepts/02a-std_atmos-P/index.html
    expected_interpolated_density_3000 = 0.9091
    assert np.isclose(earth.rhoa(3000), expected_interpolated_density_3000, atol=1e-3)
    expected_interpolated_density_25000 = 0.0395
    assert np.isclose(earth.rhoa(25000), expected_interpolated_density_25000, atol=1e-3)
    


def test_extrapolation(earth):
    expected_extrapolated_density = 0.000003
    assert np.isclose(earth.rhoa(89700), expected_extrapolated_density, atol=1e-3)
    assert np.isclose(earth.rhoa(110000), 0.0000001, atol=1e-3)
    # Test negative altitude.
    assert np.isclose(earth.rhoa(-1000), 1.347, atol=1e-3)

if __name__  == '__main__':
    pytest.main([__file__])