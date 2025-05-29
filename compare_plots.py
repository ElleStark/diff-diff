# Script to plot comparisons of specific gravity results under different conditions
# Elle Stark May 2025 

import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('data/SG_20C_2sigma_0RH_0.95vapeff.pkl', 'rb') as f:
    sg_0RH_set = pickle.load(f)

with open('data/SG_20C_2sigma_30RH_0.95vapeff.pkl', 'rb') as f1:
    sg_30RH_set = pickle.load(f1)

with open('data/SG_20C_2sigma_90RH_0.95vapeff.pkl', 'rb') as f2:
    sg_90RH_set = pickle.load(f2)

all_times = list(sg_0RH_set.keys())
print(len(all_times))
plot_times = [all_times[1], all_times[3], all_times[20]]
r_vals = r_max = 0.06 # choose radius for computation, m
r_vals = np.linspace(-r_max, r_max, 1000)

for time in plot_times:
    sg0RH = sg_0RH_set[time]
    sg30RH = sg_30RH_set[time]
    sg90RH = sg_90RH_set[time]
    plt.plot(r_vals*100, sg0RH, 'k', linestyle='--', label='0% RH')
    plt.plot(r_vals*100, sg30RH, 'k', label='30% RH')
    plt.plot(r_vals*100, sg90RH, 'k', linestyle=':', label='90% RH')
    # plt.ylim(mol_mass_helium*mol_m3/(mol_mass_air*mol_m3), (c0_ace*mol_mass_acetone+(mol_mass_CDA*mol_m3)*(1-mfrac_ace))/(mol_mass_air*mol_m3))
    plt.legend()
    plt.ylim(0.99, 1.1)
    plt.xlim(-4, 4)
    plt.xlabel('radial distance (cm)')
    plt.ylabel('specific gravity')
    plt.savefig(f'figures/SG_0to30to90RH_2sigma_t{round(time, 3)}.png', dpi=600)
    plt.show()


