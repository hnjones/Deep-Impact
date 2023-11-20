"""Module to calculate the damage and impact risk for given scenarios"""
import pandas as pd
import os

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
    blat = lat
    blon = lon
    damrad = [5000.] * len(pressures)

    return blat, blon, damrad


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

    return (pd.DataFrame({'postcode': '', 'probability': 0}, index=range(1)),
            {'mean': 0., 'stdev': 0.})
