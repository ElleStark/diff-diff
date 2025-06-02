# Script to analytically estimate molecular diffusion of acetone and helium over time
# toward an understanding of differential diffusion in wind tunnel experiments
# Elle Stark May 2025

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from scipy.special import erf

###### USER INPUTS ######

# Lab conditions
temp_degC = 20  # temperature in Celsius
rel_humidity = 90  # percent indoor relative humidity (varies a lot each day, I pulled from https://www.accuweather.com/en/us/university-of-colorado-at-boulder/80309/current-weather/107865_poi)

# Define time and length scales of interest
adv_tscale = 3  # advective timescale for defining time vector for computing concentrations
times = np.linspace(0, adv_tscale, 300)
r_max = 0.05 # choose largest radius for computation, m
r_vals = np.linspace(-r_max, r_max, 1000)
tube_d = 0.01  # exit tube diameter, m

##### END USER INPUTS #####

def main():

    # Print selected parameters for calculations
    print(f'Temperature: {temp_degC} deg C')
    print('--------------------------')

    # temperatures in Kelvin for computations
    temp_degK = temp_degC + 273.15
    basetemp_K = 273.15  # standard temperature for tables of chemical properties

    # atmostpheric pressure
    p_atm = 0.83  # ambient pressure in bar for Boulder, CO
    p_atm_base = 1.01325  # standard pressure, in bar (equal to 1 atm), for tables of chemical properties

    # acetone vapor pressure calcs: Antoine equation parameters (from https://webbook.nist.gov/cgi/inchi?ID=C67641&Mask=4&Type=ANTOINE&Plot=on#ANTOINE)
    # for log10(P)=A-(B/(T+C)) with P in bar and T in degrees C
    acetone_A = 4.42448
    acetone_B = 1312.253
    acetone_C = 240.71

    # molecular weights in g/mol
    mol_mass_acetone = 58.08
    mol_mass_helium = 4.003 
    mol_mass_CDA = 28.96 # cold dry air from building supply
    mol_mass_water = 18.0153

    # molar mass of ambient air less than CDA due to water vapor/ relative humidity
    # Antoine equation constants for water (from NIST https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on#ANTOINE)
    water_A = 5.40221
    water_B = 1838.675
    water_C = 241.413
    psat_water = 10**(water_A-water_B/(temp_degC+water_C))
    p_water = rel_humidity/100 * psat_water 
    mfrac_water = p_water / p_atm
    mol_mass_air = mfrac_water * mol_mass_water + (1-mfrac_water) * mol_mass_CDA 

    # compute 95% saturated acetone vapor pressure (bar) from Antoine equation
    ace_sat_eff = 0.95  # saturation efficiency (fraction of saturated concentration)
    p_ace = ace_sat_eff * 10**(acetone_A-acetone_B/(temp_degC+acetone_C)) 
    print(f'Acetone vapor pressure (bar): {round(p_ace, 3)}')
    mfrac_ace = p_ace / p_atm

    # parameters for computing diffusion coefficient. D0 is diffusivity at 0 deg C and 1 atm. 
    D0_He = 6.41e-5  # m^2/s, from Boynton & Brattain International Critical Tables, Volume V, pg 62 : https://reader.library.cornell.edu/docviewer/digital?id=chla2944761_2174#page/72/mode/1up 
    d_exp_He = 1.75  # temperature ratio exponent for ocmputing diffusivity under different conditions
    D0_ace = 1.09e-5  # m^2/s, from Perry's Chemical Engineers handbook, section 2 pg 328: https://students.aiu.edu/submissions/profiles/resources/onlineBook/z5y2E6_Perry-s_Chemical_Engineers-_Handbook.pdf 
    d_exp_ace = 2  # temperature ratio exponent for computing diffusivity under different conditions
    D0_water = 2.2e-5  # m^2/s, from Perry's Chemical Engineers handbook, section 2 pg 328
    d_exp_water = 1.75

    # compute diffusion coefficient for given conditions
    D_acetone = D0_ace * (temp_degK / basetemp_K)**d_exp_ace * (p_atm_base / p_atm)
    D_helium = D0_He * (temp_degK / basetemp_K)**d_exp_He * (p_atm_base / p_atm)
    D_water = D0_water * (temp_degK / basetemp_K)**d_exp_water * (p_atm_base / p_atm)
    print(f'Acetone diffusivity: {round(D_acetone, 7)} m^2/s')
    print(f'Helium diffusivity: {round(D_helium, 7)} m^2/s')
    print(f'water vapor diffusivity: {round(D0_water, 7)} m^2/s')

    # ideal gas law for computing number of moles per m3 of volume
    # PV = nRT, so n = (pV) / (RT)
    p = 1 # volume in m3
    R = 0.0000821  # (m3 atm) / (mol K)
    mol_m3 = (p_atm/1.01325) / (R*temp_degK)

    # initial volume ratios and mass such that at t=0, SG=1
    mfrac_CDA = (mol_mass_air - mol_mass_acetone*mfrac_ace-mol_mass_helium+mol_mass_helium*mfrac_ace) / (mol_mass_CDA-mol_mass_helium)
    mfrac_hel = 1 - mfrac_CDA - mfrac_ace
    print(f'eqn check: Mair={mol_mass_air}, RHS={mol_mass_acetone*mfrac_ace+mol_mass_helium*mfrac_hel+mol_mass_CDA*mfrac_CDA}')
    pct_He = mfrac_hel/(mfrac_hel+mfrac_CDA)
    pct_CDA = mfrac_CDA/(mfrac_hel+mfrac_CDA)
    print(f'acetone molar fraction:{round(mfrac_ace, 3)}')
    print(f'helium molar fraction:{round(mfrac_hel, 3)}')
    print(f'air molar fraction:{round(mfrac_CDA, 3)}')
    print(f'percent He in carrier gas: {round(pct_He*100, 2)}%')
    print(f'percent air in carrier gas: {round(pct_CDA*100, 2)}%')

    # volumetric computations for carrier gas components (for actual experiments rather than analytical diffusion quantification)
    windspeed = 0.10  # m/s
    tube_area = np.pi*tube_d**2 / 4
    source_rate = windspeed * tube_area * 1000 * 60  # volumetric flowrate (L/min)
    He_rate = pct_He * source_rate
    CDA_rate = pct_CDA * source_rate
    print(f'source flow rate: {round(source_rate, 3)} LPM')
    print(f'volumetric rate for He (isokinetic): {round(He_rate, 3)} LPM')
    print(f'volumetric rate for Air (isokinetic): {round(CDA_rate, 3)} LPM')


    # start time, sec, for point source such that x standard deviations are within exit tube diameter at t=0
    n_sigma = 2  # how many standard deviations to set within tube diameter
    ace_start = (tube_d/(2*n_sigma/2))**2 / (-2 * D_acetone)  
    hel_start = (tube_d/(2*n_sigma/2))**2 / (-2 * D_helium)

    # c0_ace = mol_mass_acetone * mfrac_ace * mol_m3 / 1000  # C0 in kg/m3
    # c0_hel = mol_mass_helium * mfrac_hel * mol_m3 / 1000
    c0_ace = mfrac_ace * mol_m3  # C0 in mol/m3
    print(f'Acetone C0: {round(c0_ace, 3) } mol/m3; {round(c0_ace*mol_mass_acetone/1000, 3)} kg/m3')
    c0_hel = mfrac_hel * mol_m3
    print(f'Helium C0: {round(c0_hel, 3)} mol/m3; {round(c0_hel*mol_mass_helium/1000, 3)} kg/m3')
    cinf_water = mfrac_water * mol_m3
    print(f'Water Cinf: {round(cinf_water, 3)} mol/m3')

    # initial number of moles such that at t=0, SG=1
    m_ace = c0_ace * (4*np.pi*D_acetone*(-ace_start))**(3/2)
    m_hel = c0_hel * (4*np.pi*D_helium*(-hel_start))**(3/2)

    # initialize dictionaries for C values at selected times
    C_ace_set = dict()
    C_hel_set = dict()
    C_water_set = dict()
    sg_set = dict()

    # Analytical solution to diffusion INTO a sphere (see Crank, 1973: The Mathematics of Diffusion)
    def concentration_profile(r, t, R, D, C_s, N_terms):
        sum_series = np.zeros_like(r)
        for n in range(1, N_terms + 1):
            term = ((-1)**n / n) * np.sin(n * np.pi * r / R) * np.exp(-D * (n * np.pi / R)**2 * t)
            sum_series += term
        C = C_s * (1 + (2 * R / (np.pi * r)) * sum_series)
        return C

    for t in times:
        # Compute evolving concentration (mol/m3) for acetone
        C_acetone = m_ace / (4*np.pi*D_acetone*(t-ace_start))**(3/2)*np.exp(-r_vals**2/(4*D_acetone*(t-ace_start)))
        # Compute evolving concentration (mol/m3) for helium
        C_helium = m_hel / (4*np.pi*D_helium*(t-hel_start))**(3/2)*np.exp(-r_vals**2/(4*D_helium*(t-hel_start)))
        C_hel_set[t] = C_helium
        C_ace_set[t] = C_acetone

        # Compute specific gravity at each time
        v_hel = C_helium / (mol_m3)
        v_ace = C_acetone / (mol_m3)
        totalair_vfrac = 1-v_hel - v_ace
        # compute concentration of water vapor using simplified 1D approximation of diffusion equation for constant C at the tube boundary and a 'boundary' at the center
        C_water = np.zeros(C_helium.shape)
        # C_water = cinf_water/2 * (1+erf((abs(r_vals)-tube_d/2)/np.sqrt(4*D_water*t)))
        # C_water[C_water>cinf_water] = cinf_water
        if np.sqrt(D_water*t) > (0.8*tube_d/2):
            C_water = concentration_profile(abs(r_vals), t, tube_d/2, D_water, cinf_water, 1000)
        else:
            # C_water = cinf_water * erf((tube_d/2-abs(r_vals))/(2 * np.sqrt(D_water*t)))
            C_water = cinf_water/2 * (1+erf((abs(r_vals)-tube_d/2)/np.sqrt(4*D_water*t)))
        C_water_set[t] = C_water

        water_vfrac = C_water / mol_m3
        cda_vfrac = 1 - water_vfrac - v_ace - v_hel
        # print(f'water fraction at center: {water_vfrac[500]}')
        sg_set[t] = (C_acetone * mol_mass_acetone / 1000 + C_helium * mol_mass_helium / 1000 + mol_mass_CDA*(cda_vfrac)*(mol_m3)/1000 + mol_mass_water*water_vfrac*mol_m3/1000) / (mol_mass_air*mol_m3/1000)

    # Save sets of concentration and specific gravity if desired
    file_ids = f'{temp_degC}C_{n_sigma}sigma_{rel_humidity}RH_{ace_sat_eff}vapeff'
    with open(f'data/SG_{file_ids}.pkl', 'wb') as f1:
        pickle.dump(sg_set, f1)

    with open(f'data/Cace_{file_ids}.pkl', 'wb') as f2:
        pickle.dump(C_ace_set, f2)

    with open(f'data/Chel_{file_ids}.pkl', 'wb') as f3:
        pickle.dump(C_hel_set, f3)


    # Plot comparing acetone and helium curves at a few times of interest
    # t=0, should be nearly identical, but scaled, curves with 2 sigma = 1 cm diameter
    plot_times = [times[5], times[100], times[-1]]
    #plot_times = times[0:10]
    for time in plot_times:
        # Conservation of mass check: multiply by 4pir^2
        # plt.plot(r_vals, C_ace_set[time]*4*np.pi*(r_vals)**2, color='#5CB7A5', label='acetone')
        # plt.plot(r_vals, C_hel_set[time]*4*np.pi*(r_vals)**2, color='#E86F44', label='helium', linestyle='dashed')
        # check water concentration
        plt.plot(r_vals, C_water_set[time])
        plt.show()

        plt.plot(r_vals, C_ace_set[time], color='#5CB7A5', label='acetone')
        plt.plot(r_vals, C_hel_set[time], color='#E86F44', label='helium', linestyle='dashed')
        plt.vlines(-0.005, 0, 100, color='k', linestyles='dotted', label='exit tube diameter', linewidth=0.5)
        plt.vlines(0.005, 0, 100, color='k', linestyles='dotted', linewidth=0.5)
        plt.ylabel(r'molar concentration (mol/$\mathrm{m}^3$)')
        plt.xlabel('Radial distance (m)')
        plt.xlim(-r_max, r_max)
        # plt.xlim(-3, 3)
        plt.legend(loc='upper right')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.ylim(0, max((C_ace_set[time][int(500)]), (C_hel_set[time][int(500)])))
        # plt.ylim(0,max(max(C_ace_set[time]*4*np.pi*(r_vals)**2), max(C_hel_set[time]*4*np.pi*(r_vals)**2)) )
        # plt.title(f't={round(time, 1)}')
        plt.savefig(f'figures/molC2sigma_{round(temp_degC, 0)}C_ace_hel_t{round(time, 3)}_{round(rel_humidity, 0)}RH_{ace_sat_eff}sateff.png', dpi=300)
        plt.show()

    # Compute and plot the specific gravity as a function of r for a few times of interest
    for time in plot_times:
        # v_hel = C_hel_set[time] / (mol_m3)
        # v_ace = C_ace_set[time] / (mol_m3)
        # #print(max(v_hel[:]))
        # cda_vfrac = 1-v_hel - v_ace
        # # print((air_vfrac[500]))
        # sg = (C_ace_set[time] * mol_mass_acetone / 1000 + C_hel_set[time] * mol_mass_helium / 1000 + mol_mass_CDA*(cda_vfrac)*(mol_m3)/1000) / (mol_mass_air*mol_m3/1000)


        sg = sg_set[time]
        plt.plot(r_vals*100, sg, 'k')
        # plt.ylim(mol_mass_helium*mol_m3/(mol_mass_air*mol_m3), (c0_ace*mol_mass_acetone+(mol_mass_CDA*mol_m3)*(1-mfrac_ace))/(mol_mass_air*mol_m3))
        # plt.ylim(0.2, 1.2)
        plt.ylim(0.99, 1.1)
        plt.xlim(-4, 4)
        plt.xlabel('radial distance (cm)')
        plt.ylabel('specific gravity')
        plt.savefig(f'figures/SG_{round(temp_degC, 0)}C_2sigma_t{round(time, 3)}_{round(rel_humidity, 0)}RH_Crank_{ace_sat_eff}sateff.png', dpi=300)
        plt.show()


    #### ANIMATION OF RESULTS ####

    # # Precompute specific gravity over all timesteps
    # SG_all = []
    # for t in times:
    #     v_hel = C_hel_set[t] / mol_m3
    #     v_ace = C_ace_set[t] / mol_m3
    #     air_vfrac = 1 - v_hel - v_ace
    #     sg = (C_ace_set[t] * mol_mass_acetone / 1000 + 
    #           C_hel_set[t] * mol_mass_helium / 1000 + 
    #           rho_air * air_vfrac) / rho_air
    #     SG_all.append(sg)

    # # Set up figure and axis
    # fig, ax = plt.subplots()
    # line, = ax.plot([], [], 'k')
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(0.99, 1.1)
    # ax.set_xlabel('Radial distance (cm)')
    # ax.set_ylabel('Specific gravity')
    # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # # Initialization function
    # def init():
    #     line.set_data([], [])
    #     time_text.set_text('')
    #     return line, time_text

    # # Animation function
    # def update(frame):
    #     x = r_vals * 100
    #     y = SG_all[frame]
    #     line.set_data(x, y)
    #     time_text.set_text(f'Time: {times[frame]:.3f} s')
    #     return line, time_text

    # # Create animation
    # ani = animation.FuncAnimation(fig, update, frames=len(times),
    #                               init_func=init, blit=True, interval=20)

    # # Save as MP4 using ffmpeg
    # ani.save('figures/specific_gravity_evolution.mp4', fps=int(len(times)/adv_tscale/10), writer='ffmpeg', dpi=300)

if __name__=='__main__':
    main()
