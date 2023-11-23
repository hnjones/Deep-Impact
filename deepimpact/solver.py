"""
This module contains the atmospheric entry solver class
for the Deep Impact project
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

__all__ = ["Planet"]


class Planet:
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(
        self,
        atmos_func="exponential",
        atmos_filename=os.sep.join(
            (os.path.dirname(__file__), "..", "resources", "AltitudeDensityTable.csv")
        ),
        Cd=1.0,
        Ch=0.1,
        Q=1e7,
        Cl=1e-3,
        alpha=0.3,
        Rp=6371e3,
        g=9.81,
        H=8000.0,
        rho0=1.2,
    ):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        try:
            # set function to define atmoshperic density
            if atmos_func == "exponential":
                self.rhoa = lambda z: rho0 * np.exp(-z / H)
            elif atmos_func == "tabular":
                self.read_csv()
                self.rhoa = lambda x: self.interpolate_density(x)
            elif atmos_func == "constant":
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0

    def rk4_step(self, f, y, t, dt):
        """RK4."""
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, y + k1 / 2)
        k3 = dt * f(t + dt / 2, y + k2 / 2)
        k4 = dt * f(t + dt, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve_atmospheric_entry(
        self,
        radius,
        velocity,
        density,
        strength,
        angle,
        init_altitude=100e3,
        dt=0.25,
        radians=False,
    ):
        if not radians:
            angle = np.radians(angle)

        # def simple_equations_of_motion(t, y):
        #     v, m, theta, z, x, r = y
        #     rho_a = self.rhoa(z)
        #     A = np.pi * r**2

        #     dvdt = (-self.Cd * rho_a * A * v**2) / (2 * m)
        #     dmdt = 0
        #     dthetadt = 0
        #     dzdt = -v * np.sin(theta)
        #     dxdt = v * np.cos(theta)
        #     drdt = 0

        #     return np.array([dvdt, dmdt, dthetadt, dzdt, dxdt, drdt])

        def equations_of_motion(t, y):
            v, m, theta, z, x, r = y
            rho_a = self.rhoa(z)
            A = np.pi * r**2

            dvdt = (-self.Cd * rho_a * A * v**2) / (2 * m) + self.g * np.sin(theta)
            dmdt = (-self.Ch * rho_a * A * v**3) / (2 * self.Q)
            dthetadt = (
                (self.g * np.cos(theta)) / v
                - (self.Cl * rho_a * A * v) / (2 * m)
                - (v * np.cos(theta)) / (self.Rp + z)
            )
            dzdt = -v * np.sin(theta)
            dxdt = (v * np.cos(theta)) / (1 + z / self.Rp)
            drdt = (
                np.sqrt((7 / 2) * self.alpha * (rho_a / density)) * v
                if rho_a * v**2 > strength
                else 0
            )

            return np.array([dvdt, dmdt, dthetadt, dzdt, dxdt, drdt])

        y0 = np.array(
            [
                velocity,
                density * (4 / 3) * np.pi * radius**3,
                angle,
                init_altitude,
                0,
                radius,
            ]
        )
        t = 0
        results = []
        fragmented = False
        user_time_elapsed = 0.0  # Initialize the user-specified time elapsed counter
        results.append([t] + list(y0))

        while True:
            # dt_actual = min(dt - user_time_elapsed, 0.01)
            dt_actual = min(dt, 0.01)

            y0 = self.rk4_step(equations_of_motion, y0, t, dt_actual)
            t += dt_actual
            # user_time_elapsed = dt
            user_time_elapsed += dt_actual

            if y0[1] <= 0 or y0[3] <= 0 or y0[0] < 0:
                break
            if len(results) > 0 and y0[3] > results[-1][4]:
                break
            # if len(results) > 0:
            #     altitude_change = abs(y0[3] - results[-1][4])
            #     if altitude_change < 1:

            # Check for height changes when the cumulative time meets or
            # exceeds theuser-defined dt
            if user_time_elapsed >= dt:
                # If the previous result exists and the height change
                # is less than 1, the simulation is stopped
                if len(results) > 0 and abs(y0[3] - results[-1][4]) < 1:
                    break
                results.append([t] + list(y0))
                user_time_elapsed = 0.0  # Reset the user-specified time elapsed counter

            ram_pressure = self.rhoa(y0[3]) * y0[0] ** 2
            if ram_pressure > strength:
                fragmented = True
            elif fragmented and ram_pressure <= strength:
                fragmented = False

        result_df = pd.DataFrame(
            results,
            columns=[
                "time",
                "velocity",
                "mass",
                "angle",
                "altitude",
                "distance",
                "radius",
            ],
        )

        # Converts the angle column in the result from radians to degrees
        result_df["angle"] = np.degrees(result_df["angle"])

        return result_df

    # def calculate_energy(self, result):
    #     # Calculate the kinetic energy
    #     kinetic_energy = 0.5 * result['mass'] * result['velocity']**2

    #     # Convert kinetic energy from Joules to kilotons of TNT
    #     kinetic_energy_kt = kinetic_energy / 4.184e12

    #     # Calculate the energy difference between successive steps, prepend the first value to maintain array size
    #     energy_diff = np.diff(kinetic_energy_kt, prepend=kinetic_energy_kt[0])

    #     # Calculate the altitude difference between successive steps
    #     altitude_diff = np.diff(result['altitude'], prepend=result['altitude'][0])

    #     # Replace any zero altitude differences with a small value to avoid division by zero
    #     small_value = 1e-6  # This can be adjusted as needed
    #     altitude_diff[altitude_diff == 0] = small_value

    #     # Calculate dedz, convert from per meter to per kilometer
    #     dedz = energy_diff / (altitude_diff / 1000)

    #     # Update or create the 'dedz' column
    #     if 'dedz' in result.columns:
    #         result['dedz'] = dedz
    #     else:
    #         result.insert(len(result.columns), 'dedz', dedz)

    #     return result

    # def analyse_outcome(self, result):
    #     """
    #     Inspect a pre-found solution to calculate the impact and airburst stats

    #     Parameters
    #     ----------
    #     result : DataFrame
    #         pandas dataframe with velocity, mass, angle, altitude, horizontal
    #         distance, radius and dedz as a function of time

    #     Returns
    #     -------
    #     outcome : Dict
    #         dictionary with details of the impact event, which should contain
    #         the key:
    #             ``outcome`` (which should contain one of the
    #             following strings: ``Airburst`` or ``Cratering``),
    #         as well as the following 4 keys:
    #             ``burst_peak_dedz``, ``burst_altitude``,
    #             ``burst_distance``, ``burst_energy``
    #     """

    #     outcome = {'outcome': 'Unknown',
    #                'burst_peak_dedz': 0.,
    #                'burst_altitude': 0.,
    #                'burst_distance': 0.,
    #                'burst_energy': 0.}
    #     # Check if the DataFrame is empty
    #     if result.empty:
    #         return outcome

    #     # Find the index of the maximum energy deposition rate
    #     max_dedz_idx = result['dedz'].idxmax()
    #     max_dedz = result.loc[max_dedz_idx, 'dedz']

    #     # Check if the max energy deposition occurs at an altitude above 0
    #     if result.loc[max_dedz_idx, 'altitude'] > 0:
    #         outcome['outcome'] = 'Airburst'
    #         outcome['burst_peak_dedz'] = max_dedz
    #         outcome['burst_altitude'] = result.loc[max_dedz_idx, 'altitude']
    #         outcome['burst_distance'] = result.loc[max_dedz_idx, 'distance']
    #         outcome['burst_energy'] = result.loc[max_dedz_idx, 'dedz']
    #     else:
    #         outcome['outcome'] = 'Cratering'
    #         # For cratering, determine the specifics at the point of ground impact

    #     return outcome

    # def read_csv(self):
    #     with open(self.atmos_filename, 'r') as file:
    #         next(file)  # Skip the header line
    #         data = np.loadtxt(file)
    #         self.altitudes = data[:, 0]
    #         self.densities = data[:, 1]
    #         self.interpolator = interp1d(self.altitudes, self.densities, kind='cubic', fill_value="extrapolate")

    # def interpolate_density(self, x):
    #     return self.interpolator(x)
