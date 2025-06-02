# script to compute and plot terminal velocity for a range of densities

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex']= True

# Define parameters
c_d = 0.47  # drag coefficient for a sphere
r = 0.005  # radius of sphere in meters
g = 9.81  # m2/s acceleration due to gravity
temp_degC = 20  # temperature in Celsius
rel_humidity = 30  # percent indoor relative humidity (varies, I pulled from https://www.accuweather.com/en/us/university-of-colorado-at-boulder/80309/current-weather/107865_poi)

# molar mass of ambient air less than CDA due to water vapor/ relative humidity
# Antoine equation constants for water (from NIST https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on#ANTOINE)
R = 0.0000821  # (m3 atm) / (mol K)
p_atm = 0.83  # ambient pressure in bar for Boulder, CO
mol_m3 = (p_atm/1.01325) / (R*(temp_degC+273.15))
mol_mass_CDA = 28.96 # cold dry air from building supply
mol_mass_water = 18.0153
water_A = 5.40221
water_B = 1838.675
water_C = 241.413
psat_water = 10**(water_A-water_B/(temp_degC+water_C))
p_water = rel_humidity/100 * psat_water 
mfrac_water = p_water / p_atm
mol_mass_air = mfrac_water * mol_mass_water + (1-mfrac_water) * mol_mass_CDA
rho_air = mol_mass_air * mol_m3 / 1000  # kg/m3  

# Range of sphere densities
SG_range = np.linspace(1, 1.15, 300)
rho_s_range = SG_range * rho_air

# compute terminal velocity values
t_vel = np.sqrt(np.pi * r * g * (rho_s_range - rho_air) / (c_d * rho_air))

# plot terminal velocity vs density
plt.plot(SG_range, t_vel, 'k-')
plt.xlabel('specific gravity')
plt.ylabel('terminal velocity (m/s)')
plt.title(r'Terminal velocity for 1 cm sphere in $20^\mathrm{o}$ C, 30\% RH')
plt.savefig('figures/termvel_30RH_20C.png', dpi=300)
plt.show()

# compute time to reach terminal velocity for a few mass values
SG_range2 = np.linspace(1, 1.15, 15)
rho_s_range2 = SG_range2 * rho_air
mass_range = rho_s_range2 * 4/3 * np.pi * r**3
mass_air = rho_air * 4/3 * np.pi * r**3
vel_dict = {}
vel_list = []
times = np.linspace(0, 3, 1000)
delta_t = times[1] - times[0]
for mass in mass_range:
    t=0
    vel_list = []
    vel = 0  # initial velocity in m/s
    F_gb = (mass - mass_air) * g
    while t < len(times):
        F_d = 1/2 * rho_air * vel**2 * c_d * np.pi * r**2
        F_net = F_gb - F_d
        acc = F_net / mass
        vel_new = vel + acc * (delta_t)
        vel_list.append(vel_new)
        vel = vel_new
        t+=1
    vel_dict[str(mass)] = vel_list[:]
    del vel_list[:]

# vel_lists = sorted(vel_dict.items())
# t_list, vel_values = zip(*vel_lists)
plt.plot(times, vel_dict[str(mass_range[0])], label=f'SG={SG_range2[0]}')
plt.plot(times, vel_dict[str(mass_range[5])], label=f'SG={round(SG_range2[5], 2)}')
plt.plot(times, vel_dict[str(mass_range[-1])], label=f'SG={SG_range2[-1]}')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.legend()
plt.show()



