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


######## Constants for computations ########

basetemp_K = 273.15  # standard temperature for tables of chemical properties

# atmostpheric pressure
p_atm = 0.83  # ambient pressure in bar for Boulder, CO
p_atm_base = 1.01325  # standard pressure, in bar (equal to 1 atm), for tables of chemical properties

# acetone vapor pressure calcs: Antoine equation parameters (from https://webbook.nist.gov/cgi/inchi?ID=C67641&Mask=4&Type=ANTOINE&Plot=on#ANTOINE)
# for log10(P)=A-(B/(T+C)) with P in bar and T in degrees C
acetone_A = 4.42448
acetone_B = 1312.253
acetone_C = 240.71
# alternate constants for pressure in mmHg from Physical and Chemical Equilibrium for Chemical Engineers (2012): https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1 
# acetone_A = 7.02447
# acetone_B = 1161
# acetone_C = 224

# molar mass of ambient air less than CDA due to water vapor/ relative humidity
# Antoine equation constants for water (from NIST https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on#ANTOINE)
water_A = 5.40221
water_B = 1838.675
water_C = 241.413

# molecular weights in g/mol
mol_mass_acetone = 58.08
mol_mass_helium = 4.003 
mol_mass_CDA = 28.96 # cold dry air from building supply
mol_mass_water = 18.0153

# parameters for computing diffusion coefficient. D0 is diffusivity at 0 deg C and 1 atm. 
D0_He = 6.41e-5  # m^2/s, from Boynton & Brattain International Critical Tables, Volume V, pg 62 : https://reader.library.cornell.edu/docviewer/digital?id=chla2944761_2174#page/72/mode/1up 
d_exp_He = 1.75  # temperature ratio exponent for ocmputing diffusivity under different conditions
D0_ace = 1.09e-5  # m^2/s, from Perry's Chemical Engineers handbook, section 2 pg 328: https://students.aiu.edu/submissions/profiles/resources/onlineBook/z5y2E6_Perry-s_Chemical_Engineers-_Handbook.pdf 
d_exp_ace = 2  # temperature ratio exponent for computing diffusivity under different conditions
D0_water = 2.2e-5  # m^2/s, from Perry's Chemical Engineers handbook, section 2 pg 328
d_exp_water = 1.75

# Ideal gas constant
R = 0.0000821  # (m3 atm) / (mol K)

#############################################


class Experiment:

    def __init__(self, temp, RH, times, r_vals, tube_d, eff):
        self.temp_C = temp
        self.RH = RH
        self.times = times
        self.r_vals = r_vals
        self.tube_d = tube_d
        self.tube_r = tube_d/2
        self.vapeff = eff

        # Compute phycial chemistry properties for above conditions
        # temperatures in Kelvin for computations
        self.temp_degK = self.temp_C + 273.15

        # Print selected parameters for calculations
        INFO(f'Temperature: {self.temp_C} deg C')
        INFO('--------------------------')

        # compute molar mass of ambient air
        psat_water = 10**(water_A-water_B/(self.temp_C+water_C))
        p_water = self.RH/100 * psat_water 
        mfrac_water = p_water / p_atm
        self.mol_mass_air = mfrac_water * mol_mass_water + (1-mfrac_water) * mol_mass_CDA 
        
        # compute 95% saturated acetone vapor pressure (bar) from Antoine equation
        ace_sat_eff = self.vapeff  # saturation efficiency (fraction of saturated concentration)
        p_ace = ace_sat_eff * 10**(acetone_A-acetone_B/(self.temp_C+acetone_C)) 
        # p_ace = p_ace * 0.00133322  # need to convert to bar from mmHg if using alternate Antoine constants!
        INFO(f'Acetone vapor pressure (bar): {round(p_ace, 3)}')
        self.mfrac_ace = p_ace / p_atm

        # compute diffusion coefficient for given conditions
        self.D_acetone = D0_ace * (self.temp_degK / basetemp_K)**d_exp_ace * (p_atm_base / p_atm)
        self.D_helium = D0_He * (self.temp_degK / basetemp_K)**d_exp_He * (p_atm_base / p_atm)
        self.D_water = D0_water * (self.temp_degK / basetemp_K)**d_exp_water * (p_atm_base / p_atm)
        INFO(f'Acetone diffusivity: {round(self.D_acetone, 7)} m^2/s')
        INFO(f'Helium diffusivity: {round(self.D_helium, 7)} m^2/s')
        INFO(f'water vapor diffusivity: {round(self.D_water, 7)} m^2/s')

        # ideal gas law for computing number of moles per m3 of volume
        # PV = nRT, so n = (pV) / (RT)
        V = 1 # volume in m3
        self.mol_m3 = (p_atm/1.01325) * V / (R*self.temp_degK)


    # Analytical solution to diffusion INTO a sphere (see Crank, 1973: The Mathematics of Diffusion)
    def concentration_profile(r, t, R, D, C_s, N_terms):
        sum_series = np.zeros_like(r)
        for n in range(1, N_terms + 1):
            term = ((-1)**n / n) * np.sin(n * np.pi * r / R) * np.exp(-D * (n * np.pi / R)**2 * t)
            sum_series += term
        C = C_s * (1 - (2 * R / (np.pi * r)) * sum_series)
        return C