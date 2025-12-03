"""
10-kVA, GFL
===========

This example simulates a 10-kVA grid-following (GFL) converter connected to an L filter
and a strong grid. The control system includes a phase-locked loop (PLL) to synchronize
with the grid, a current reference generator, and a PI-based current controller.

"""

# %%
from motulator.grid import control, model, utils

# %%
# Compute base values based on the nominal values.

nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
base = utils.BaseValues.from_nominal(nom)

# %%
# Configure the system model.

# Filter and grid
# ac_filter = model.LFilter(L_f=0.15 * base.L, L_g=0.05 * base.L, R_g=0.005 * base.Z)
ac_filter = model.LFilterLCLGrid(
    L_f=0.15 * base.L,
    C_g=0.08 * base.C,
    L_g1=0.05 * base.L,
    L_g2=0.2 * base.L,
    R_g2=0.005 * base.Z,
)
ac_source = model.ThreePhaseSource(w_g=base.w, e_g=base.u)
converter = model.VoltageSourceConverter(u_dc=650)
mdl = model.GridConverterSystem(converter, ac_filter, ac_source)

# %%
# Configure the control system.

inner_ctrl = control.CurrentVectorController(i_max=1.3 * base.i, L=0.15 * base.L)
ctrl = control.GridConverterControlSystem(inner_ctrl)

# %%
# Set the time-dependent reference and disturbance signals.

# Set the active and reactive power references
# ctrl.set_power_ref(lambda t: (t > 0.02) * 5e3)
ctrl.set_power_ref(
    lambda t: ((t > 0.2) / 3 + (t > 0.5) / 3 + (t > 0.8) / 3 - (t > 1.2)) * nom.P
)
ctrl.set_reactive_power_ref(lambda t: (t > 0.04) * 0)

# Uncomment lines below to simulate an unbalanced fault (add negative sequence)
# from math import pi
# mdl.ac_source.e_g = 0.75 * base.u
# mdl.ac_source.e_g_neg = 0.25 * base.u
# mdl.ac_source.phi_neg = -pi / 3

# %%
# Create the simulation object, simulate, and plot the results in per-unit values.

sim = model.Simulation(mdl, ctrl)
res = sim.simulate(t_stop=1.6)
utils.plot_control_signals(
    res, base, t_lims=(0.0, 1.5), y_lims=[(-0.7, 1.5), (-0.1, 1.5), (0.0, 1.5)]
)
utils.plot_grid_waveforms(res, base, plot_pcc_voltage=False)


# Uncomment line below to plot locus of the grid voltage space vector
# utils.plot_voltage_vector(res, base)
