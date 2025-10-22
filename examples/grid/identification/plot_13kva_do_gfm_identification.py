"""
12.5-kVA, DO-GFM, converter output admittance identification
============================================================

This example demonstrates converter output admittance identification using a 12.5-kVA
disturbance-observer-based grid-forming (DO-GFM) converter.

"""

# %%
from motulator.grid import control, model, utils
from motulator.grid.utils._identification import (
    IdentificationCfg,
    plot_identification,
    plot_vector_diagram,
    run_identification,
)

# %%
# Compute base values based on the nominal values.

nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
base = utils.BaseValues.from_nominal(nom)

# %%
# Configure the identification.

identification_cfg = IdentificationCfg(
    abs_u_e=0.01 * base.u,
    f_start=1,
    f_stop=10e3,
    n_freqs=100,
    multiprocess=True,
    T_s=1 / 10e3,
    # Uncomment the row below to save identification results in "project root"/data
    # filename="do-gfm_admittance",
    filetype="csv",
)

# %%
# Configure the system model.

ac_filter = model.LFilter(L_f=0.15 * base.L, L_g=0.74 * base.L)
ac_source = model.ThreePhaseSource(w_g=base.w, e_g=base.u)
converter = model.VoltageSourceConverter(u_dc=650)
mdl = model.GridConverterSystem(
    converter, ac_filter, ac_source, delay=identification_cfg.delay
)

# %%
# Configure the control system.

inner_ctrl = control.ObserverBasedGridFormingController(
    i_max=1.3 * base.i,
    L=0.15 * base.L,
    R_a=0.2 * base.Z,
    k_v=1,
    alpha_o=base.w,
    u_nom=base.u,
    w_nom=base.w,
    T_s=identification_cfg.T_s,
)
ctrl = control.GridConverterControlSystem(inner_ctrl)

ctrl.set_power_ref(0.5 * base.p)
ctrl.set_ac_voltage_ref(base.u)

# %%
# Run the identification and plot results.

if __name__ == "__main__":
    res = run_identification(identification_cfg, mdl, ctrl)
    plot_identification(res, plot_style="re_im")
    plot_vector_diagram(res, base)
