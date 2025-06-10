"""Tool for converter output admittance identification."""

from typing import Any

from motulator.grid import control, model, utils
from motulator.grid.utils._identification import (
    IdentificationCfg,
    plot_identification,  # noqa: F401
    plot_vector_diagram,  # noqa: F401
    run_identification,
)


# %%
def setup_parameter_sweep() -> Any:
    """
    Configure the parameter sweep.

    sweep : dict[Literal["p_g", "q_g", "L_f"], array_like, shape (N,)] | None, optional
        If specified, run a parameter sweep using the given values. For example,
        `sweep = {"p_g": np.array([-1.0, -0.5, 0.0, 0.5, 1.0])*base.p}` specifies a
        sweep for active power reference ranging from -1 p.u. to 1 p.u. with 0.5 p.u.
        increments. More than one parameter can be specified, but the array sizes need
        to be the same. Defaults to None.

    """
    raise NotImplementedError
    # sweep = ({"p_g": np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * base.p},)


# %%
def setup_identification() -> tuple[
    IdentificationCfg,
    model.GridConverterSystem,
    control.GridConverterControlSystem,
    utils.BaseValues,
]:
    """Configure the identification."""

    # Compute base values based on the nominal values.
    nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
    base = utils.BaseValues.from_nominal(nom)

    # Configure the identification
    identification_cfg = IdentificationCfg(
        op_point={"p_g": 0.87 * base.p, "q_g": 0.5 * base.p, "v_c": base.u},
        abs_u_e=0.01 * base.u,
        f_start=1,
        f_stop=10e3,
        n_freqs=100,
        multiprocess=True,
        T_s=1 / 10e3,
        delay=1,
        k_comp=1.5,
        # filename=None,
        filename="obs_f1-10k_n100log_p0.87_delay1_Lg0.74",
        # filename="gfl_f1-10k_n100log_p0.5_q0.5_delay1_Lf0.05",
        filetype="csv",
    )

    # Configure the system model.
    ac_filter = model.LFilter(L_f=0.15 * base.L, L_g=0.74 * base.L)
    ac_source = model.ThreePhaseSourceWithSignalInjection(w_g=base.w, e_g=base.u)
    converter = model.VoltageSourceConverter(u_dc=650)

    # Create system model
    mdl = model.GridConverterSystem(
        converter, ac_filter, ac_source, delay=identification_cfg.delay
    )

    # Configure the control system.

    # # GFL
    # inner_ctrl = control.CurrentVectorController(
    #     i_max=1.5 * base.i,
    #     L=0.15 * base.L,
    #     u_nom=base.u,
    #     w_nom=base.w,
    #     T_s=identification_cfg.T_s,
    #     k_comp=identification_cfg.k_comp,
    # )

    # Observer GFM
    inner_ctrl = control.ObserverBasedGridFormingController(
        i_max=1.3 * base.i,
        L=0.15 * base.L,
        R_a=0.2 * base.Z,
        k_v=1,
        alpha_o=base.w,
        u_nom=base.u,
        w_nom=base.w,
        T_s=identification_cfg.T_s,
        k_comp=identification_cfg.k_comp,
    )

    ctrl = control.GridConverterControlSystem(inner_ctrl)
    return identification_cfg, mdl, ctrl, base


# %%
if __name__ == "__main__":
    # Run the identification
    cfg, mdl, ctrl, base = setup_identification()
    res = run_identification(cfg, mdl, ctrl)

    # plot_identification(res, plot_style="re_im")
    # plot_vector_diagram(res, base)
