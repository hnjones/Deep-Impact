"""Module dealing with postcode information."""

import numpy as np
import pandas as pd

__all__ = ['GeospatialLocator', 'great_circle_distance']


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """
    # Converting latitudes and longitudes from degrees to radians
    latlon1_rad = np.radians(latlon1)
    latlon2_rad = np.radians(latlon2)

    # Earth's radius in meters
    R = 6371000

    # Trigonometric values
    sin_lat1 = np.sin(latlon1_rad[:, 0])[:, np.newaxis]
    sin_lat2 = np.sin(latlon2_rad[:, 0])
    cos_lat1 = np.cos(latlon1_rad[:, 0])[:, np.newaxis]
    cos_lat2 = np.cos(latlon2_rad[:, 0])
    delta_lon = latlon2_rad[:, 1] - latlon1_rad[:, 1][:, np.newaxis]

    # Applying the great circle formula
    cos_d = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * np.cos(delta_lon)

    # Clipping to avoid precision issues
    cos_d = np.clip(cos_d, -1, 1) 

    # Compute distance
    distance = R * np.arccos(cos_d)

    return distance


class GeospatialLocator(object):
    """
    Class to interact with a postcode database file and a population grid file.
    """

    def __init__(self, postcode_file='',
                 census_file='',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .asc file containing census data on a
            latitude-longitude grid.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """

        self.postcode_file = postcode_file
        self.census_file = census_file
        self.norm = norm
        self.postcodes = self.load_postcode_data()
        self.census = self.load_census_data() 

    def load_postcode_data(self):
        # Load postcode data from CSV
        if self.postcode_file:
            df = pd.read_csv(self.postcode_file)
            return df
        return pd.DataFrame()
    
    def get_postcodes_by_radius(self, X, radii):
        """
        Return postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.


        Examples
        --------

        >>> locator = GeospatialLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [1.5e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773),
                                            [1.5e3, 4.0e3])
        """
        if self.postcodes.empty:
            return [[] for _ in radii]

        result = []
        for radius in radii:
            # Calculating distances to all postcodes
            distances = self.norm(self.postcodes[['Latitude', 'Longitude']].values, [X])
            
            # Filter postcodes within the radius
            within_radius = self.postcodes[distances[:, 0] <= radius]
            result.append(within_radius['Postcode'].tolist())

        return result
    
    def load_census_data(self):
        with open(self.census_file, 'r') as file:
            # Headers
            ncols = int(file.readline().split()[1])
            nrows = int(file.readline().split()[1])
            nodata_value = float(file.readline().split()[1])
            # Skipping the array labels
            _ = file.readline()
            _ = file.readline()
            _ = file.readline()

            # Read the data
            data = np.loadtxt(file, dtype=float)

            # Reshaping data into 3 arrays as they are stacked one after the other
            data = data.reshape((-1, nrows, ncols))

            # Updating the population to 0 for missing values
            data[2][data[2] == -9999] = 0

            latitude = data[0].flatten()
            longitude = data[1].flatten()
            population = data[2].flatten()

            df = pd.DataFrame({
                'Latitude': latitude,
                'Longitude': longitude,
                'Population': population
            })

        return df
        
    def get_population_by_radius(self, X, radii):
        """
        Return the population within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list
            Contains the population closer than the elements of radii to
            the location X. Output should be the same shape as the radii array.

        Examples
        --------
        >>> loc = GeospatialLocator()
        >>> loc.get_population_by_radius((51.4981, -0.1773), [1e2, 5e2, 1e3])

        """
        # Calculate distances from X to each point in the census data
        self.census['distance'] = self.norm(self.census[
            ['Latitude', 'Longitude']].values, [X])[:, 0]

        populations_by_radius = []
        for radius in radii:
            # Sum population for points within the radius
            total_population = self.census[
                self.census['distance'] <= radius]['Population'].sum()
            populations_by_radius.append(total_population)

        return populations_by_radius