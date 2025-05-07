"""Functions and classes for converter output admittance identification."""

import copy
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from os.path import join
from time import time
from typing import Any

import numpy as np
from scipy.io import savemat
from scipy.signal.windows import blackman

from motulator.common.utils import empty_array
from motulator.grid import control, model, utils


# %%
def setup_identification() -> tuple[
    "IdentificationCfg", model.GridConverterSystem, control.GridConverterControlSystem
]:
    """Configure the identification."""

    # Compute base values based on the nominal values.
    nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
    base = utils.BaseValues.from_nominal(nom)

    # Configure the identification
    identification_cfg = IdentificationCfg(
        op_point={"p_g": 0.5 * base.p, "q_g": 0.5 * base.p, "v_c": base.u},
        abs_u_e=0.01 * base.u,
        f_start=1,
        f_stop=10e3,
        n_freqs=100,
        multiprocess=True,
        spacing="log",
        n_periods=4,
        T_s=1 / 10e3,
        N_eval=10,
        use_window=False,
        delay=0,
        k_comp=0.5,
        # filename=None,
        # filename="obs_f1-10k_n100log_p0.5_delay0",
        filename="gfl_f1-10k_n100log_p0.5_q0.5_delay0_nowindow",
        plot_style=None,
    )

    # Configure the system model.
    ac_filter = model.LFilter(L_f=0.15 * base.L)
    ac_source = model.ThreePhaseSourceWithSignalInjection(w_g=base.w, e_g=base.u)
    converter = model.VoltageSourceConverter(u_dc=650)

    # Create system model
    mdl = model.GridConverterSystem(
        converter, ac_filter, ac_source, delay=identification_cfg.delay
    )

    # Configure the control system.

    # GFL
    inner_ctrl = control.CurrentVectorController(
        L=0.15 * base.L,
        u_nom=base.u,
        w_nom=base.w,
        i_max=1.5 * base.i,
        T_s=identification_cfg.T_s,
        k_comp=identification_cfg.k_comp,
    )

    # # Observer GFM
    # inner_ctrl = control.ObserverBasedGridFormingController(
    #     L=0.15 * base.L,
    #     u_nom=base.u,
    #     w_nom=base.w,
    #     i_max=1.3 * base.i,
    #     R_a=0.2 * base.Z,
    #     k_v=1,
    #     alpha_o=base.w,
    #     T_s=identification_cfg.T_s,
    #     k_comp=identification_cfg.k_comp,
    # )

    ctrl = control.GridConverterControlSystem(inner_ctrl)

    return identification_cfg, mdl, ctrl


# %%
@dataclass
class IdentificationCfg:
    """
    Configuration parameters for converter output admittance identification.

    Parameters
    ----------
    op_point : dict[str, float]
        Dictionary object containing the reference signals in the operating point.
    abs_u_e : float
        Magnitude of the voltage excitation (V).
    f_start : float
        Starting frequency of the voltage excitation (Hz).
    f_stop : float
        End frequency of the voltage excitation (Hz).
    n_freqs : int
        Number of frequencies for measurement.
    spacing : str, optional
        Whether to use logarithmic "log" or linear "lin" spacing for creating
        the array of measurement frequencies. The default is "log".
    manual_freqs : ndarray | None, optional
        Manually specified array of frequencies (Hz) to measure admittance at.
        If set to None, f_start, f_stop and n_freqs parameters are used to
        create the array of frequencies. The default is None.
    t0 : float, optional
        Stop time for initial simulating to the operating point (s). Should be
        large enough to reach steady-state. The default is 0.1.
    t1 : float, optional
        Additional simulation time for reaching steady-state during signal
        injection (s). The default is 0.02.
    T_s : float, optional
        Sampling period of the control system (s). The default is 1/10e3.
    N_eval : int, optional
        Number of evenly spaced data points the solver should return for each
        controller sampling period. The default is 10.
    n_periods : int, optional
        Number of excitation signal periods to use for calculating the DFT. The
        default is 4.
    multiprocess : bool, optional
        If set to True, multiprocessing.Pool() is used to run the
        identification using parallel threads. The default is True.
    plot_style : str, optional
        Set this variable to plot either the real and imaginary parts of the
        admittance ("re_im") or the magnitude and phase ("bode"). Can also be
        set to None to disable plotting. The default is "re_im".
    filename : str, optional
        If given, the identification result is saved in
        */matfiles/{date}_{time}_{filename}.mat where * is the project
        root directory. The default is None.
    delay : int, optional
        Number of samples for modeling the computational delay. The default
        is 1.
    k_comp : float, optional
        Compensation factor for the delay effect on the converter output
        voltage vector angle. The default is 1.5.
    use_window : bool, optional
        Whether to use window function for calculating DFT. Defaults to True.

    """

    op_point: dict[str, float]
    abs_u_e: float
    f_start: float
    f_stop: float
    n_freqs: int
    spacing: str = "log"
    manual_freqs: np.ndarray | None = None
    t0: float = 0.1
    t1: float = 0.02
    T_s: float = 1 / 10e3
    N_eval: int = 10
    n_periods: int = 4
    multiprocess: bool = True
    plot_style: str | None = "re_im"
    filename: str | None = None
    delay: int = 1
    k_comp: float = 1.5
    use_window: bool = True

    def __post_init__(self) -> None:
        # Create array of frequencies if not specified
        if self.manual_freqs is None:
            if self.spacing == "lin":
                self.freqs = np.linspace(self.f_start, self.f_stop, self.n_freqs)
            else:
                self.freqs = np.geomspace(self.f_start, self.f_stop, self.n_freqs)
        else:
            self.freqs = self.manual_freqs
        # Increase excitation signal amplitude with the frequency
        self.amplitudes = self.abs_u_e * np.linspace(1.0, 5.0, np.size(self.freqs))


@dataclass
class IdentificationResults:
    """Container for identification results"""

    f_e: np.ndarray = field(default_factory=empty_array)
    Y_dd: np.ndarray = field(default_factory=empty_array)
    Y_qd: np.ndarray = field(default_factory=empty_array)
    Y_dq: np.ndarray = field(default_factory=empty_array)
    Y_qq: np.ndarray = field(default_factory=empty_array)


# %%


def save_mat(data: IdentificationResults, filename: str) -> None:
    """Save the identification results in a .mat-file."""

    # Convert the data class to dict
    data_dict = dict(data.__dict__.items())
    # Create the file path
    timestamp = datetime.now().strftime("%Y%m%d_%H.%M_")
    filepath = join("matfiles", timestamp + filename + ".mat")

    try:
        savemat(filepath, data_dict)
        print(f"Data successfully exported to {timestamp + filename}")
    except Exception as error:
        print(f"Error saving data: {str(error)}")


def dft(cfg: IdentificationCfg, u: np.ndarray, f_e: float) -> complex:
    """
    Single-frequency discrete Fourier transform.

    Calculates the frequency component y at frequency f_e from input
    signal u, using the discrete Fourier transform algorithm.

    """

    n = int(cfg.n_periods * cfg.N_eval / (f_e * cfg.T_s))
    if cfg.use_window:
        u = u[-n:] * blackman(n, False)
    else:
        u = u[-n:]
    y = (
        2
        / n
        * np.sum(u * np.exp(-2j * np.pi * f_e * cfg.T_s / cfg.N_eval * np.arange(n)))
    )
    return y


def copy_state(sim: model.Simulation) -> tuple[Any, Any]:
    """Make a copy of the simulation state."""

    mdl = copy.deepcopy(sim.mdl)
    ctrl = copy.deepcopy(sim.ctrl)
    return mdl, ctrl


def pre_process(
    cfg: IdentificationCfg,
    mdl: model.GridConverterSystem,
    ctrl: control.GridConverterControlSystem,
) -> model.Simulation:
    """Simulate the system to the desired operating point."""

    # Set appropriate references
    if isinstance(ctrl.dc_bus_voltage_ctrl, control.DCBusVoltageController):
        ctrl.set_dc_bus_voltage_ref(cfg.op_point["u_dc"])
    else:
        ctrl.set_power_ref(cfg.op_point["p_g"])
    if isinstance(ctrl.inner_ctrl, control.CurrentVectorController):
        ctrl.set_reactive_power_ref(cfg.op_point["q_g"])
    if isinstance(ctrl.inner_ctrl, control.ObserverBasedGridFormingController):
        ctrl.set_ac_voltage_ref(cfg.op_point["v_c"])

    # Create Simulation object and simulate
    sim = model.Simulation(mdl, ctrl, show_progress=False)
    _ = sim.simulate(t_stop=cfg.t0)
    return sim


def identify(
    cfg: IdentificationCfg, sim: model.Simulation, i: int, f_e: float
) -> list[Any]:
    """Calculate the output admittance at a single frequency."""

    # 1) d-axis injection
    mdl, ctrl = copy_state(sim)
    mdl.ac_source.f_e = f_e
    mdl.ac_source.u_ed = cfg.amplitudes[i]
    # Set new stop time and simulate
    t_stop = mdl.t0 + cfg.t1 + cfg.n_periods / f_e
    sim_d = model.Simulation(mdl, ctrl, show_progress=False)
    res_d = sim_d.simulate(t_stop=t_stop, N_eval=cfg.N_eval)

    # Transform the voltage and current to synchronous coordinates and
    # calculate the DFT
    u_g1 = np.conj(res_d.mdl.ac_source.exp_j_theta_g) * res_d.mdl.ac_filter.u_g_ab
    u_gd1 = dft(cfg, u_g1.real, f_e)
    u_gq1 = dft(cfg, u_g1.imag, f_e)
    i_g1 = np.conj(res_d.mdl.ac_source.exp_j_theta_g) * res_d.mdl.ac_filter.i_g_ab
    i_gd1 = dft(cfg, i_g1.real, f_e)
    i_gq1 = dft(cfg, i_g1.imag, f_e)

    # 2) q-axis injection
    mdl, ctrl = copy_state(sim)
    mdl.ac_source.f_e = f_e
    mdl.ac_source.u_eq = cfg.amplitudes[i]
    sim_q = model.Simulation(mdl, ctrl, show_progress=False)
    res_q = sim_q.simulate(t_stop=t_stop, N_eval=cfg.N_eval)

    # DFT
    u_g2 = np.conj(res_q.mdl.ac_source.exp_j_theta_g) * res_q.mdl.ac_filter.u_g_ab
    u_gd2 = dft(cfg, u_g2.real, f_e)
    u_gq2 = dft(cfg, u_g2.imag, f_e)
    i_g2 = np.conj(res_q.mdl.ac_source.exp_j_theta_g) * res_q.mdl.ac_filter.i_g_ab
    i_gd2 = dft(cfg, i_g2.real, f_e)
    i_gq2 = dft(cfg, i_g2.imag, f_e)

    # # Print DFT coefficients for debugging
    # print(
    #     f"f_e: {f_e:8.1f} u_gd1: {np.abs(u_gd1):5.2f} u_gq1: {np.abs(u_gq1):5.2f} "
    #     + f"u_gd2: {np.abs(u_gd2):5.2f} u_gq2: {np.abs(u_gq2):5.2f}"
    # )

    # Calculate the elements of the output admittance matrix
    I = np.array([[i_gd1, i_gd2], [i_gq1, i_gq2]])  # noqa: E741
    U = np.array([[u_gd1, u_gd2], [u_gq1, u_gq2]])
    inv_U = np.linalg.inv(U)
    Y_c = -1 * I @ inv_U

    Y_dd = Y_c[0, 0]
    Y_qd = Y_c[0, 1]
    Y_dq = Y_c[1, 0]
    Y_qq = Y_c[1, 1]
    return [i, f_e, Y_dd, Y_qd, Y_dq, Y_qq]


def post_process(results: list[list[Any]]) -> IdentificationResults:
    """Transform the results to ndarray format."""
    results_array = np.vstack(results)
    res = IdentificationResults(
        f_e=np.real(results_array[:, 1]),
        Y_dd=results_array[:, 2],
        Y_qd=results_array[:, 3],
        Y_dq=results_array[:, 4],
        Y_qq=results_array[:, 5],
    )
    return res


def run_identification() -> None:
    """Entrypoint for running the identification."""
    cfg, mdl, ctrl = setup_identification()
    results = []
    sim_op = pre_process(cfg, mdl, ctrl)
    t_start = time()

    def custom_error_callback(error):
        print(f"Error during multiprocessing: {str(error)}")

    def collect_result(result):
        results.append(result)

    if cfg.multiprocess:
        # Create the multiprocessing pool using all available CPUs
        with mp.Pool(mp.cpu_count()) as pool:
            async_results = []
            for i, f_e in enumerate(cfg.freqs):
                async_result = pool.apply_async(
                    identify,
                    args=[cfg, sim_op, i, f_e],
                    error_callback=custom_error_callback,
                    callback=collect_result,
                )
                async_results.append(async_result)
            # Wait for all tasks to complete
            [res.get() for res in async_results]
        # Sort the results list along the first element (index i)
        results.sort(key=lambda x: x[0])
    else:
        # Run only in single thread
        for i, f_e in enumerate(cfg.freqs):
            result = identify(cfg, sim_op, i=i, f_e=f_e)
            collect_result(result)

    print(f"Execution time: {(time() - t_start):.2f} s")
    data = post_process(results)
    if cfg.filename is not None:
        save_mat(data, cfg.filename)
    # if cfg.plot_style is not None:
    #     plot_identification(data, cfg.plot_style)


if __name__ == "__main__":
    # Run the identification
    run_identification()
