"""
12.5-kVA, RFPSC-GFM
===================

This example simulates grid-forming (GFM) converter using reference-feedforward
power-synchronization control (RFPSC). The converter is connected to a weak grid.

"""

# %%
from math import pi  # noqa: F401

from motulator.grid import control, model, utils

# %%
# Compute base values based on the nominal values.

nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
base = utils.BaseValues.from_nominal(nom)

# %%
# Configure the system model.

ac_filter = model.LFilter(
    L_f=0.15 * base.L, R_f=0.0 * base.Z, L_g=0.5 * base.L, R_g=0.0 * base.Z
)
ac_source = model.ThreePhaseSource(w_g=base.w, e_g=base.u)
converter = model.VoltageSourceConverter(u_dc=650)
mdl = model.GridConverterSystem(converter, ac_filter, ac_source)

# %%
# Configure the control system.

# Control configuration parameters
inner_ctrl = control.VirtualSynchronousMachineController(
    u_nom=base.u,
    w_nom=base.w,
    i_nom=base.i,
    i_max=1.3 * base.i,
    L=0.15 * base.L,
    # R=0.05 * base.Z,
    sigma=0.05,
    H=1.0,
    # k_D=500,
)
ctrl = control.GridConverterControlSystem(inner_ctrl)

# %%
# Set the references for converter output voltage magnitude and active power.

# Converter output voltage magnitude reference
ctrl.set_ac_voltage_ref(base.u)

# Active power reference
ctrl.set_power_ref(
    lambda t: ((t > 1.0) / 3 + (t > 2.0) / 3 + (t > 3.0) / 3 - (t > 4.0)) * nom.P
)
# ctrl.set_power_ref(lambda t: (t > 0.0) * 0.5 * nom.P)

mdl.ac_source.w_g = lambda t: (1 - (t > 7.0) * 0.02) * base.w
# times = np.asarray([0, 2.0, 2.5, 8.0])
# values = np.asarray([base.w, base.w, 1.01 * base.w, 1.01 * base.w])
# mdl.ac_source.w_g = SequenceGenerator(times, values)

# %%
# Create the simulation object, simulate, and plot the results in per-unit values.

sim = model.Simulation(mdl, ctrl)
res = sim.simulate(t_stop=10.0)
# utils.plot_control_signals(res, base)
utils.plot_control_signals(
    res, base, y_lims=[(-1.0, 1.2), (-1.0, 1.5), (0.0, 1.25), (0.97, 1.01)]
)
utils.plot_grid_waveforms(res, base)
