"""Module to calculate the damage and impact risk for given scenarios"""
from collections import Counter
import os
import math
import pandas as pd
import folium
import numpy as np
from folium.plugins import HeatMap
#from .locator import GeospatialLocator
import deepimpact
__all__ = ['damage_zones', 'impact_risk']


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------

    >>> import deepimpact
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> deepimpact.damage_zones(outcome, 52.79, -2.95, 135,
                                pressures=[1e3, 3.5e3, 27e3, 43e3])
    """

    # Replace this code with your own. For demonstration we just
    # return lat, lon and a radius of 5000 m for each pressure
    burst_altitude = outcome['burst_altitude']
    burst_energy = outcome['burst_energy']
    burst_distance = outcome.get('burst_distance', 0)  # default to 0 if not found

    # Calculate the surface zero point
    blat, blon = find_destination(lat, lon, bearing, burst_distance)

    # Calculate damage radii for each pressure threshold
    damrad = calculate_damage_radius(pressures, burst_altitude, burst_energy)
    
    return blat, blon, damrad


def find_destination(lat, lon, bearing, distance):
    """
    Find the destination point given starting latitude, longitude, bearing and distance.
    Assumes a spherical Earth.
    """
    R = 6371000  # Radius of the Earth in meters
    bearing = math.radians(bearing)  # Convert bearing to radians

    phi1 = math.radians(lat)  # Current lat point converted to radians
    lambda1 = math.radians(lon)  # Current long point converted to radians

    sin_phi2 = math.sin(phi1) * math.cos(distance/R) + math.cos(phi1) * math.sin(distance/R) * math.cos(bearing)
    lat2 = math.asin(sin_phi2)

    tan_lambda = (math.sin(bearing) * math.sin(distance/R) * math.cos(phi1))/ (math.cos(distance/R) - math.sin(phi1) * math.sin(lat2))
    lon2 = math.atan(tan_lambda) + lambda1

    return math.degrees(lat2), math.degrees(lon2)

def calculate_damage_radius(target_pressures, z_b, E_k):
    def p(r, z_b, E_k):
        term1 = 3e11 * np.power((r**2 + z_b**2) / np.power(E_k, 2/3), -1.3)
        term2 = 2e7 * np.power((r**2 + z_b**2) / np.power(E_k, 2/3), -0.57)
        return term1 + term2

    def dp(r, z_b, E_k):
        term1 = -1.3 * 3e11 * np.power((r**2 + z_b**2) / np.power(E_k, 2/3), -2.3) * (2 * r / np.power(E_k, 2/3))
        term2 = -0.57 * 2e7 * np.power((r**2 + z_b**2) / np.power(E_k, 2/3), -1.57) * (2 * r / np.power(E_k, 2/3))
        return term1 + term2

    def newtons_method(target_pressure, z_b, E_k, initial_guess= 10000, tolerance=1e-6, max_iterations=100):
        r = initial_guess

        for iteration in range(max_iterations):
            pressure_value = p(r, z_b, E_k)
            pressure_diff = pressure_value - target_pressure

            if abs(pressure_diff) < tolerance:
                print(f"Converged after {iteration} iterations.")
                return r

            derivative = dp(r, z_b, E_k)

            if derivative == 0:
                # Avoid division by zero
                print("Derivative is zero. Stopping.")
                break

            r -= pressure_diff / derivative

        raise ValueError("Newton's method did not converge within the maximum number of iterations.")

    radii=[]
    for i in target_pressures:
        radii.append(newtons_method(i,z_b,E_k))
    flipped_list = radii[::-1]
    return flipped_list


def impact_risk(planet,
                impact_file=os.sep.join((os.path.dirname(__file__),
                                         '..', 'resources',
                                         'impact_parameter_list.csv')),
                pressure=30.e3, nsamples=None):
    """
    Perform an uncertainty analysis to calculate the probability for
    each affected UK postcode and the total population affected.

    Parameters
    ----------
    planet: deepimpact.Planet instance
        The Planet instance from which to solve the atmospheric entry

    impact_file: str
        Filename of a .csv file containing the impact parameter list
        with columns for 'radius', 'angle', 'velocity', 'strength',
        'density', 'entry latitude', 'entry longitude', 'bearing'

    pressure: float
        A single pressure at which to calculate the damage zone for each impact

    nsamples: int or None
        The number of iterations to perform in the uncertainty analysis.
        If None, the full set of impact parameters provided in impact_file
        is used.

    Returns
    -------
    probability: DataFrame
        A pandas DataFrame with columns for postcode and the
        probability the postcode was inside the blast radius.
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.
    """


    data = pd.read_csv(impact_file)
    data=data.iloc[:nsamples]
    postcodes_all=[]
    population_all=[]
    for i in range(data.shape[0]):
        print(f"data{i}strat")
        result = planet.solve_atmospheric_entry(radius=data.loc[i,'radius'],
                                                angle=data.loc[i,'angle'],
                                                strength=data.loc[i,'strength'],
                                                density=data.loc[i,'density'],
                                                velocity=data.loc[i,'velocity'])
        result = planet.calculate_energy(result)
        outcome = planet.analyse_outcome(result)
        
        
        blast_lat, blast_lon, damage_rad=damage_zones(outcome,lat=data.loc[i,'entry latitude'],
                                                      lon=data.loc[i,'entry longitude'],
                                                      bearing=data.loc[i,'bearing'],
                                                      pressures=[pressure])

        locators = deepimpact.GeospatialLocator()
        postcodes = locators.get_postcodes_by_radius((blast_lat, blast_lon),radii=damage_rad)
        population = locators.get_population_by_radius((blast_lat, blast_lon),radii=damage_rad)
        postcodes_all=postcodes_all + postcodes[-1]
        population_all.append(population[-1])
        print(f"data{i}end")
    element_counts = Counter(postcodes_all)
    postcodes_pos = {key: count / data.shape[0] for key, count in element_counts.items()}

    return (pd.DataFrame(postcodes_pos, index=range(1)),
            {'mean': np.mean(population_all), 'stdev': np.std(population_all)})

def impact_risk_plot(probability, population):

    """
    Plot the probability of postcodes using heatmap.

    Parameters
    ----------
    probability: DataFrame
        A pandas DataFrame with columns for postcode and the
        probability the postcode was inside the blast radius.
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.

    Returns
    -------
    weighted_heatmap:html
        A html saved in local to show the probability of different postcode 
        impacted by the scenario.

    """ 
    data_post = pd.read_csv(os.sep.join((os.path.dirname(__file__),
                                         '..', 'resources',
                                         'full_postcodes.csv')))
    probability_new = pd.DataFrame({'Postcode':probability.columns.tolist(),
                                    'probability':probability.iloc[0].tolist()})
    data_with_weights =  pd.merge(probability_new, data_post[['Postcode', 'Latitude', 'Longitude']],
                                  on='Postcode', how='left')
    m = folium.Map(location=[data_with_weights.Latitude[0],
                             data_with_weights.Longitude[0]], zoom_start=13)
    HeatMap(data_with_weights[['Latitude', 'Longitude','probability']].values.tolist(),
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}, radius=15, blur=10).add_to(m)
    m.save("weighted_heatmap.html")
