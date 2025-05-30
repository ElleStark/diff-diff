# Class for constructing, computing, and plotting the spherical diffusion of acetone, helium, and water vapor
# Elle Stark, June 2025

import numpy as np
import matplotlib.pyplot as plt
import logging 

logger = logging.getLogger('DiffDiff')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warning
DEBUG = logger.debug

class Experiment:
    # Constants for computations
    basetemp_K = 273.15  # standard temperature for tables of chemical properties

    # atmostpheric pressure
    p_atm = 0.83  # ambient pressure in bar for Boulder, CO
    p_atm_base = 1.01325  # standard pressure, in bar (equal to 1 atm), for tables of chemical properties

    # acetone vapor pressure calcs: Antoine equation parameters (from https://webbook.nist.gov/cgi/inchi?ID=C67641&Mask=4&Type=ANTOINE&Plot=on#ANTOINE)
    # for log10(P)=A-(B/(T+C)) with P in bar and T in degrees C
    acetone_A = 4.42448
    acetone_B = 1312.253
    acetone_C = 240.71

    def __init__(self, temp, RH, times, r_vals, tube_d):
        self.temp_C = temp
        self.RH = RH
        self.times = times
        self.r_vals = r_vals
        self.tube_d = tube_d
        self.tube_r = tube_d/2

        # Compute phycial chemistry properties for above conditions
        # temperatures in Kelvin for computations
        self.temp_degK = self.temp_C + 273.15


        

    # Analytical solution to diffusion INTO a sphere (see Crank, 1973: The Mathematics of Diffusion)
    def concentration_profile(r, t, R, D, C_s, N_terms):
        sum_series = np.zeros_like(r)
        for n in range(1, N_terms + 1):
            term = ((-1)**n / n) * np.sin(n * np.pi * r / R) * np.exp(-D * (n * np.pi / R)**2 * t)
            sum_series += term
        C = C_s * (1 - (2 * R / (np.pi * r)) * sum_series)
        return C