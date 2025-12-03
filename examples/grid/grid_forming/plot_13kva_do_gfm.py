"""
12.5-kVA, DO-GFM
================

This example simulates a 12.5-kVA disturbance-observer-based grid-forming (DO-GFM)
converter, connected to a weak grid. The converter output voltage and the active power
are directly controlled. Grid synchronization is provided by the disturbance observer.
A transparent current controller is included for current limitation.

"""

# %%
import numpy as np  # noqa: F401

from motulator.common.utils._utils import SequenceGenerator  # noqa: F401
from motulator.grid import control, model, utils

# %%
# Compute base values based on the nominal values.

nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
base = utils.BaseValues.from_nominal(nom)

# %%
# Configure the system model.

ac_filter = model.LFilter(L_f=0.15 * base.L, L_g=0.74 * base.L)
# ac_filter = model.LFilterLCLGrid(
#     L_f=0.15 * base.L,
#     C_g=0.08 * base.C,
#     L_g1=0.05 * base.L,
#     L_g2=0.2 * base.L,
#     R_g2=0.005 * base.Z,
# )
# ac_filter = model.LFilter(
#     L_f=0.15 * base.L, R_f=0.5, L_g=0.4 * base.L, R_g=0.05 * base.Z
# )
ac_source = model.ThreePhaseSource(w_g=base.w, e_g=base.u)
converter = model.VoltageSourceConverter(u_dc=650)
mdl = model.GridConverterSystem(converter, ac_filter, ac_source)

# %%
# Configure the control system.

inner_ctrl = control.ObserverBasedGridFormingController(
    i_max=1.3 * base.i,
    L=0.15 * base.L,
    # R=0.05 * base.Z,
    R_a=0.05 * base.Z,
    u_nom=base.u,
    w_nom=base.w,
    s_base=base.p,
    H=1.0,
    sigma=0.05,
)
ctrl = control.GridConverterControlSystem(inner_ctrl)

# %%
# Set the references for converter output voltage magnitude and active power.

# Converter output voltage magnitude reference
ctrl.set_ac_voltage_ref(base.u)
# ctrl.set_power_ref(
#     lambda t: ((t > 0.2) / 3 + (t > 0.5) / 3 + (t > 0.8) / 3 - (t > 1.2)) * nom.P
# )
ctrl.set_power_ref(
    lambda t: ((t > 1.0) / 3 + (t > 2.0) / 3 + (t > 3.0) / 3 - (t > 4.0)) * nom.P
)
# ctrl.set_power_ref(lambda t: (t > 0.0) * 0.5 * nom.P)

mdl.ac_source.w_g = lambda t: (1 - (t > 7.0) * 0.02) * base.w
# times = np.asarray([0, 2.0, 2.5, 8.0])
# values = np.asarray([base.w, base.w, 1.01 * base.w, 1.01 * base.w])
# mdl.ac_source.w_g = SequenceGenerator(times, values)

# Uncomment line below to simulate operation in rectifier mode
# ctrl.ext_ref.p_g = lambda t: ((t > 0.2) - (t > 0.7) * 2 + (t > 1.2)) * nom.P

# Uncomment lines below to simulate a grid voltage sag with constant ref.p_g
# mdl.ac_filter.L_g = 0
# mdl.ac_source.e_g = lambda t: (1 - (t > 0.2) * 0.8 + (t > 1) * 0.8) * base.u
# ctrl.ext_ref.p_g = lambda t: nom.P

# %%
# Create the simulation object, simulate, and plot the results in per-unit values.

sim = model.Simulation(mdl, ctrl)
res = sim.simulate(t_stop=10.0)
utils.plot_control_signals(
    res, base, y_lims=[(-1.0, 1.2), (-1.0, 1.2), (0.0, 1.25), (0.97, 1.01)]
)
# utils.plot_control_signals(res, base)
utils.plot_grid_waveforms(res, base)
