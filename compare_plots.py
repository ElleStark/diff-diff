# Script to plot comparisons of specific gravity results under different conditions
# Elle Stark May 2025 

import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('data/SG_15C_2sigma_0RH.pkl', 'rb') as f:
    sg_15c_set = pickle.load(f)

with open('data/SG_20C_2sigma_0RH.pkl', 'rb') as f1:
    sg_20c_set = pickle.load(f1)

all_times = list(sg_15c_set.keys())
plot_times = [all_times[0], all_times[10], all_times[-1]]
r_vals = r_max = 0.06 # choose radius for computation, m
r_vals = np.linspace(-r_max, r_max, 1000)

for time in plot_times:
    sg15 = sg_15c_set[time]
    sg20 = sg_20c_set[time]
    plt.plot(r_vals*100, sg15, 'k', linestyle='--', label='')
    plt.plot(r_vals*100, sg20, 'k')
    # plt.ylim(mol_mass_helium*mol_m3/(mol_mass_air*mol_m3), (c0_ace*mol_mass_acetone+(mol_mass_CDA*mol_m3)*(1-mfrac_ace))/(mol_mass_air*mol_m3))
    # plt.ylim(0.2, 1.2)
    plt.ylim(0.99, 1.1)
    plt.xlim(-4, 4)
    plt.xlabel('radial distance (cm)')
    plt.ylabel('specific gravity')
    # plt.savefig(f'figures/SG_15C_2sigma_t{round(time, 3)}.png', dpi=300)
    plt.show()


