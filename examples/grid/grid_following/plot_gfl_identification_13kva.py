"""
10-kVA converter
================

This example simulates a grid-following-controlled converter connected to an L
filter and a strong grid. The control system includes a phase-locked loop (PLL)
to synchronize with the grid, a current reference generator, and a PI-based
current controller.

"""

# %%
from types import SimpleNamespace

#import numpy as np

from motulator.grid import model, control
from motulator.grid.utils import (
    AdmittanceIdentification, AdmittanceIdentificationCfg, BaseValues,
    ACFilterPars, NominalValues, plot, plot_identification)
# from motulator.grid.utils import plot_voltage_vector

# %%
# Compute base values based on the nominal values.

nom = NominalValues(U=400, I=18, f=50, P=12.5e3)
base = BaseValues.from_nominal(nom)

# %%
# Configure the system model.

# Filter and grid
par = ACFilterPars(L_fc=.15*base.L)
ac_filter = model.ACFilter(par)
ac_source = model.SignalInjection(w_g=base.w, abs_e_g=base.u)
# Inverter with constant DC voltage
converter = model.VoltageSourceConverter(u_dc=650)

# Create system model
mdl = model.GridConverterIdentification(
    converter, ac_filter, ac_source, delay=0)
# mdl.pwm = model.CarrierComparison()  # Uncomment to enable the PWM model

# %%
# Configure the control system.
cfg = control.GridFollowingControlCfg(
    L=.2*base.L, nom_u=base.u, nom_w=base.w, max_i=1.5*base.i, T_s=.0001)
ctrl = control.GridFollowingControl(cfg)

# %%
# Configure and run the identification
identification_cfg = AdmittanceIdentificationCfg(
    op_point=SimpleNamespace(p_g=base.p, q_g=0),
    abs_u_e=.01*base.u,
    f_start=.1,
    f_stop=1/(2*cfg.T_s),
    n_freqs=10)

identification = AdmittanceIdentification(identification_cfg, mdl, ctrl)
data = identification.run()

# %%
# Plot the identification results
plot_identification(data)
