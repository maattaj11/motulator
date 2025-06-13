"""Functions and classes for converter output admittance identification."""

import multiprocessing as mp
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from os import makedirs
from os.path import join
from time import time
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.signal.windows import blackman

from motulator.common.utils._utils import empty_array, set_latex_style, set_screen_style
from motulator.grid import control, model, utils  # noqa: F401


# %%
@dataclass
class IdentificationCfg:
    """
    Configuration parameters for converter output admittance identification.

    Parameters
    ----------
    op_point : dict[Literal["p_g", "q_g", "v_c", "u_dc"], float]
        Dictionary object containing key-value pairs for reference signals in the
        operating point. Valid options for the dictionary keys are:
        - "p_g"
        - "q_g"
        - "v_c"
        - "u_dc"
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
    filename : str | None, optional
        If given, the identification result is saved in */data/{date}_{time}_{filename}
        where * is the project root directory, defaults to None. The file format is set
        by the `filetype` parameter.
    filetype : Literal["csv", "mat"], optional
        Choose the filetype for saving identification results, defaults to "csv". Valid
        options are:
        - "csv": save results in .csv-format, requires `pandas` to be installed
        - "mat": save results in MATLAB .mat-format
    delay : int, optional
        Number of samples for modeling the computational delay. The default
        is 1.
    k_comp : float, optional
        Compensation factor for the delay effect on the converter output
        voltage vector angle. The default is 1.5.
    use_window : bool, optional
        Whether to use window function for calculating DFT. Defaults to True.

    """

    op_point: dict[Literal["p_g", "q_g", "v_c", "u_dc"], float]
    abs_u_e: float
    f_start: float = 1
    f_stop: float = 10e3
    n_freqs: int = 100
    spacing: str = "log"
    manual_freqs: np.ndarray | None = None
    t0: float = 0.1
    t1: float = 0.02
    T_s: float = 1 / 10e3
    N_eval: int = 10
    n_periods: int = 4
    multiprocess: bool = True
    filename: str | None = None
    filetype: Literal["csv", "mat"] = "csv"
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
    """
    Container for identification results.

    Contains fields for excitation signal frequency `f_e`, elements of the output
    admittance matrix `[Y_dd, Y_qd; Y_dq, Y_qq]`, operating point grid current vector
    `i_g0`, grid voltage vector `e_g0` (in synchronous coordinates aligned with the
    PCC voltage), converter output filter impedance `Z_f`, and grid impedance `Z_g`.

    """

    f_e: np.ndarray = field(default_factory=empty_array)
    Y_dd: np.ndarray = field(default_factory=empty_array)
    Y_qd: np.ndarray = field(default_factory=empty_array)
    Y_dq: np.ndarray = field(default_factory=empty_array)
    Y_qq: np.ndarray = field(default_factory=empty_array)
    i_g0: complex = 0j
    e_g0: complex = 0j
    Z_f: complex = 0j
    Z_g: complex = 0j


# %%


def save_csv(data: IdentificationResults, filename: str) -> None:
    """Save the identification results in a .csv-file."""

    try:
        from pandas import DataFrame

        # Create the data directory if it doesn't exist
        makedirs("data", exist_ok=True)

        # Create the file path
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M_")
        filepath = join("data", timestamp + filename + ".csv")

        # Transform data into pandas DataFrame object
        df = DataFrame(dict(data.__dict__.items()))

        df.to_csv(filepath)
        print(f"Data successfully exported to {timestamp + filename}")

    except Exception as error:
        print(f"Error saving data: {str(error)}")


def save_mat(data: IdentificationResults, filename: str) -> None:
    """Save the identification results in a .mat-file."""

    try:
        # Create the data directory if it doesn't exist
        makedirs("data", exist_ok=True)

        # Convert the data class to dict
        data_dict = dict(data.__dict__.items())
        # Create the file path
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M_")
        filepath = join("data", timestamp + filename + ".mat")

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
    u = u[-n:] * blackman(n, False) if cfg.use_window else u[-n:]
    y = (
        2
        / n
        * np.sum(u * np.exp(-2j * np.pi * f_e * cfg.T_s / cfg.N_eval * np.arange(n)))
    )
    return y


def copy_state(sim: model.Simulation) -> tuple[Any, Any]:
    """Make a copy of the simulation state."""

    mdl = deepcopy(sim.mdl)
    ctrl = deepcopy(sim.ctrl)
    return mdl, ctrl


def pre_process(
    cfg: IdentificationCfg,
    mdl: model.GridConverterSystem,
    ctrl: control.GridConverterControlSystem,
) -> tuple[model.Simulation, list[Any], float]:
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
    res = sim.simulate(t_stop=cfg.t0)
    # utils.plot(res, base=None)

    # Calculate operating point
    exp_j_theta_g0 = res.mdl.ac_source.exp_j_theta_g[-1]
    i_g0 = res.mdl.ac_filter.i_g_ab[-1] * np.conj(exp_j_theta_g0)
    e_g0 = res.mdl.ac_filter.e_g_ab[-1] * np.conj(exp_j_theta_g0)
    u_g0 = res.mdl.ac_filter.u_g_ab[-1] * np.conj(exp_j_theta_g0)
    u_c0 = res.mdl.ac_filter.u_c_ab[-1] * np.conj(exp_j_theta_g0)

    # Align coordinates with PCC voltage vector
    phi_g = np.angle(u_g0) - np.angle(e_g0)
    i_g0 = i_g0 * np.exp(-1j * phi_g)
    e_g0 = e_g0 * np.exp(-1j * phi_g)
    u_g0 = u_g0 * np.exp(-1j * phi_g)
    u_c0 = u_c0 * np.exp(-1j * phi_g)

    Z_f = (u_c0 - u_g0) / i_g0
    Z_g = (u_g0 - e_g0) / i_g0
    operating_point = [i_g0, e_g0, Z_f, Z_g]

    return sim, operating_point, phi_g


def identify(
    cfg: IdentificationCfg, sim: model.Simulation, i: int, f_e: float, phi_g: float
) -> list[Any]:
    """Calculate the output admittance at a single frequency."""

    # 1) d-axis injection
    mdl, ctrl = copy_state(sim)
    mdl.ac_filter.f_e = f_e
    mdl.ac_filter.u_ed = cfg.amplitudes[i]
    mdl.ac_filter.phi_g = phi_g
    # mdl.ac_source.f_e = f_e
    # mdl.ac_source.u_ed = cfg.amplitudes[i]
    # Set new stop time and simulate
    t_stop = mdl.t0 + cfg.t1 + cfg.n_periods / f_e
    sim_d = model.Simulation(mdl, ctrl, show_progress=False)
    res_d = sim_d.simulate(t_stop=t_stop, N_eval=cfg.N_eval)
    # utils.plot(res_d, base=None)

    # Transform the voltage and current to synchronous coordinates and
    # calculate the DFT
    u_g1 = (
        np.exp(-1j * phi_g)
        * np.conj(res_d.mdl.ac_source.exp_j_theta_g)
        * res_d.mdl.ac_filter.u_g_ab
    )
    u_gd1 = dft(cfg, u_g1.real, f_e)
    u_gq1 = dft(cfg, u_g1.imag, f_e)
    i_g1 = (
        np.exp(-1j * phi_g)
        * np.conj(res_d.mdl.ac_source.exp_j_theta_g)
        * res_d.mdl.ac_filter.i_g_ab
    )
    i_gd1 = dft(cfg, i_g1.real, f_e)
    i_gq1 = dft(cfg, i_g1.imag, f_e)
    # plt.plot(res_d.mdl.t, u_g1.real, res_d.mdl.t, u_g1.imag)
    # plt.show()

    # 2) q-axis injection
    mdl, ctrl = copy_state(sim)
    mdl.ac_filter.f_e = f_e
    mdl.ac_filter.u_eq = cfg.amplitudes[i]
    mdl.ac_filter.phi_g = phi_g
    # mdl.ac_source.f_e = f_e
    # mdl.ac_source.u_eq = cfg.amplitudes[i]
    sim_q = model.Simulation(mdl, ctrl, show_progress=False)
    res_q = sim_q.simulate(t_stop=t_stop, N_eval=cfg.N_eval)
    # utils.plot(res_q, base=None)

    # DFT
    u_g2 = (
        np.exp(-1j * phi_g)
        * np.conj(res_q.mdl.ac_source.exp_j_theta_g)
        * res_q.mdl.ac_filter.u_g_ab
    )
    u_gd2 = dft(cfg, u_g2.real, f_e)
    u_gq2 = dft(cfg, u_g2.imag, f_e)
    i_g2 = (
        np.exp(-1j * phi_g)
        * np.conj(res_q.mdl.ac_source.exp_j_theta_g)
        * res_q.mdl.ac_filter.i_g_ab
    )
    i_gd2 = dft(cfg, i_g2.real, f_e)
    i_gq2 = dft(cfg, i_g2.imag, f_e)
    # plt.plot(res_d.mdl.t, u_g2.real, res_d.mdl.t, u_g2.imag)
    # plt.show()

    # Print DFT coefficients for debugging
    # print(
    #     f"f_e: {f_e:8.1f} u_gd1: {np.abs(u_gd1):5.2f} u_gq1: {np.abs(u_gq1):5.2f} "
    #     + f"u_gd2: {np.abs(u_gd2):5.2f} u_gq2: {np.abs(u_gq2):5.2f} "
    #     + f"i_gd1: {np.abs(i_gd1):5.2f} i_gq1: {np.abs(i_gq1):5.2f} "
    #     + f"i_gd2: {np.abs(i_gd2):5.2f} i_gq2: {np.abs(i_gq2):5.2f}"
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


def post_process(
    results: list[list[Any]], operating_point: list[Any]
) -> IdentificationResults:
    """Save the identification results and information about the operating point."""
    results_array = np.vstack(results)
    res = IdentificationResults(
        f_e=np.real(results_array[:, 1]),
        Y_dd=results_array[:, 2],
        Y_qd=results_array[:, 3],
        Y_dq=results_array[:, 4],
        Y_qq=results_array[:, 5],
        i_g0=operating_point[0],
        e_g0=operating_point[1],
        Z_f=operating_point[2],
        Z_g=operating_point[3],
    )
    return res


# %%
def run_identification(
    cfg: IdentificationCfg,
    mdl: model.GridConverterSystem,
    ctrl: control.GridConverterControlSystem,
) -> IdentificationResults:
    """Run the identification."""
    results = []
    sim_op, operating_point, phi_g = pre_process(cfg, mdl, ctrl)
    t_start = time()
    print("Start identification...")

    index = 1
    freqs = np.size(cfg.freqs)

    def collect_result(result: list[Any]) -> None:
        nonlocal index
        results.append(result)
        print(f"\rFrequencies simulated: {index}/{freqs}", end="")
        index += 1

    def custom_error_callback(error: Any) -> None:
        print(f"Error during multiprocessing: {str(error)}")

    if cfg.multiprocess:
        # Create the multiprocessing pool using all available CPUs
        with mp.Pool(mp.cpu_count()) as pool:
            async_results = []
            for i, f_e in enumerate(cfg.freqs):
                async_result = pool.apply_async(
                    identify,
                    args=[cfg, sim_op, i, f_e, phi_g],
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
            result = identify(cfg, sim_op, i=i, f_e=f_e, phi_g=phi_g)
            collect_result(result)

    print(f"\nExecution time: {(time() - t_start):.2f} s")
    data = post_process(results, operating_point)
    if cfg.filename is not None:
        if cfg.filetype == "csv":
            save_csv(data, cfg.filename)
        elif cfg.filetype == "mat":
            save_mat(data, cfg.filename)

    return data


# %%
def plot_identification(
    res: IdentificationResults,
    plot_style: Literal["bode", "re_im"] = "re_im",
    plot_passivity_index: bool = True,
    latex: bool = False,
) -> None:
    """
    Plot the identification results

    Parameters
    ----------
    res : IdentificationResults
        Should contain the results from the identification.
    plot_style : Literal["bode", "re_im"], optional
        Style for plotting of identification results, defaults to "re_im". Options are:
        - "bode": plot magnitude (in dB) and phase (in degrees) of output admittance
        - "re_im": plot real and imaginary parts of output admittance
    plot_passivity_index : bool, optional
        Plot input feedforward passivity index calculated from the identification
        results, defaults to True.
    latex : bool, optional
        Use latex for plots, defaults to False.

    """
    # ruff: noqa: PLR0915
    if latex:
        set_latex_style()
    else:
        set_screen_style()

    # First figure: elements of output admittance matrix
    if plot_style == "bode":
        fig, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(
            4, 2, sharey="row"
        )

        ax1.semilogx(res.f_e, 20 * np.log10(np.abs(res.Y_dd)))
        ax2.semilogx(
            res.f_e, np.unwrap(np.angle(res.Y_dd, deg=True), discont=180, period=360)
        )
        ax3.semilogx(res.f_e, 20 * np.log10(np.abs(res.Y_dq)))
        ax4.semilogx(
            res.f_e, np.unwrap(np.angle(res.Y_dq, deg=True), discont=180, period=360)
        )
        ax5.semilogx(res.f_e, 20 * np.log10(np.abs(res.Y_qd)))
        ax6.semilogx(
            res.f_e, np.unwrap(np.angle(res.Y_qd, deg=True), discont=180, period=360)
        )
        ax7.semilogx(res.f_e, 20 * np.log10(np.abs(res.Y_qq)))
        ax8.semilogx(
            res.f_e, np.unwrap(np.angle(res.Y_qq, deg=True), discont=180, period=360)
        )

        ax1.set_ylabel(r"$|Y_\mathrm{dd}|\ (\mathrm{dB})$")
        ax2.set_ylabel(r"$\angle Y_\mathrm{dd}\ (\mathrm{deg})$")
        ax3.set_ylabel(r"$|Y_\mathrm{dq}|\ (\mathrm{dB})$")
        ax4.set_ylabel(r"$\angle Y_\mathrm{dq}\ (\mathrm{deg})$")
        ax5.set_ylabel(r"$|Y_\mathrm{qd}|\ (\mathrm{dB})$")
        ax6.set_ylabel(r"$\angle Y_\mathrm{qd}\ (\mathrm{deg})$")
        ax7.set_ylabel(r"$|Y_\mathrm{qq}|\ (\mathrm{dB})$")
        ax8.set_ylabel(r"$\angle Y_\mathrm{qq}\ (\mathrm{deg})$")

        ax5.tick_params(axis="y", labelleft=True)
        ax6.tick_params(axis="y", labelleft=True)
        ax7.tick_params(axis="y", labelleft=True)
        ax8.tick_params(axis="y", labelleft=True)

        ymin1, ymax1 = ax1.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()
        ymin3, ymax3 = ax3.get_ylim()
        ymin4, ymax4 = ax4.get_ylim()

        ylim_magn = (min(ymin1, ymin3), max(ymax1, ymax3))
        ax1.set_ylim(ylim_magn)
        ax3.set_ylim(ylim_magn)

        ylim_phase = (min(ymin2, ymin4), max(ymax2, ymax4))
        ax2.set_ylim(ylim_phase)
        ax4.set_ylim(ylim_phase)

    else:
        fig, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(
            4, 2, sharex=True, sharey=True
        )

        ax1.semilogx(res.f_e, np.real(res.Y_dd))
        ax2.semilogx(res.f_e, np.imag(res.Y_dd))
        ax3.semilogx(res.f_e, np.real(res.Y_dq))
        ax4.semilogx(res.f_e, np.imag(res.Y_dq))
        ax5.semilogx(res.f_e, np.real(res.Y_qd))
        ax6.semilogx(res.f_e, np.imag(res.Y_qd))
        ax7.semilogx(res.f_e, np.real(res.Y_qq))
        ax8.semilogx(res.f_e, np.imag(res.Y_qq))

        ax1.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{dd}\}\ (\mathrm{S})$")
        ax2.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{dd}\}\ (\mathrm{S})$")
        ax3.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{dq}\}\ (\mathrm{S})$")
        ax4.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{dq}\}\ (\mathrm{S})$")
        ax5.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{qd}\}\ (\mathrm{S})$")
        ax6.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{qd}\}\ (\mathrm{S})$")
        ax7.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{qq}\}\ (\mathrm{S})$")
        ax8.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{qq}\}\ (\mathrm{S})$")

        ax5.tick_params(axis="y", labelleft=True)
        ax6.tick_params(axis="y", labelleft=True)
        ax7.tick_params(axis="y", labelleft=True)
        ax8.tick_params(axis="y", labelleft=True)

    ax4.set_xlabel(r"Frequency (Hz)")
    ax8.set_xlabel(r"Frequency (Hz)")

    fig.align_ylabels()
    plt.show()

    # Second figure: passivity index
    if plot_passivity_index:
        _, ax1 = plt.subplots(1, 1, figsize=(4, 3))

        Y_c = np.array([[res.Y_dd, res.Y_qd], [res.Y_dq, res.Y_qq]])
        Y_c = np.moveaxis(Y_c, -1, 0)
        v_F = 0.5 * np.min(np.linalg.eigvals(Y_c + np.matrix_transpose(Y_c.conj())), 1)

        ax1.axhline(0, linestyle="--", color="k", linewidth="1")
        ax1.semilogx(res.f_e, v_F.real)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Passivity index")

        plt.show()


def plot_vector_diagram(
    res: IdentificationResults, base: utils.BaseValues, latex: bool = False
):
    """
    Plot the converter, PCC, and grid voltage vectors in steady-state.

    Parameters
    ----------
    res : IdentificationResults
        Should contain the results from the identification.

    """

    from matplotlib.patches import Circle

    set_screen_style()

    u_g0 = res.e_g0 + res.Z_g * res.i_g0
    u_c0 = u_g0 + res.Z_f * res.i_g0

    _, ax = plt.subplots()
    ax.grid(True, zorder=0)
    circle = Circle((0, 0), 1, fill=False, edgecolor="gray", zorder=1)
    ax.add_patch(circle)

    ax.quiver(
        0,
        0,
        res.e_g0.real / base.u,
        res.e_g0.imag / base.u,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        label=r"$\boldsymbol{e}_\mathrm{g0}$",
        zorder=2,
    )
    ax.quiver(
        0,
        0,
        u_g0.real / base.u,
        u_g0.imag / base.u,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        label=r"$\boldsymbol{u}_\mathrm{g0}$",
        zorder=2,
    )
    ax.quiver(
        0,
        0,
        u_c0.real / base.u,
        u_c0.imag / base.u,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        label=r"$\boldsymbol{u}_\mathrm{c0}$",
        zorder=2,
    )
    ax.quiver(
        0,
        0,
        res.i_g0.real / base.i,
        res.i_g0.imag / base.i,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        label=r"$\boldsymbol{i}_\mathrm{g0}$",
        zorder=2,
    )
    ticks = [-1, -0.5, 0, 0.5, 1]

    ax.set_xlabel(r"$d\ \mathrm{(p.u.)}$")
    ax.set_ylabel(r"$q\ \mathrm{(p.u.)}$")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    plt.show()
