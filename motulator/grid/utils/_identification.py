"""Functions and classes for converter output admittance identification."""

import copy
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from os.path import join
from time import time
from types import SimpleNamespace

import numpy as np
from scipy.io import savemat

from motulator.grid import model, control
from motulator.grid.utils import (
    ACFilterPars, BaseValues, NominalValues, plot, plot_identification)

# from motulator.grid.utils import plot


# %%
def setup_identification():
    """Configure the identification."""

    # Compute base values based on the nominal values.
    nom = NominalValues(U=400, I=18, f=50, P=12.5e3)
    base = BaseValues.from_nominal(nom)

    # Configure the identification
    identification_cfg = AdmittanceIdentificationCfg(
        op_point=SimpleNamespace(p_g=.5*base.p, q_g=.5*base.p),
        abs_u_e=.01*base.u,
        f_start=1,
        f_stop=5e3,  # Nyquist freq: 1/(2*cfg.T_s)
        n_freqs=100,
        multiprocess=True,
        spacing="log",
        T_eval=1/100e3,
        delay=0,
        k_comp=0.5,
        plot_style=None,
        filename="gfl_f1-5k_n100log_p0.5_q0.5_PI_test")

    # Configure the system model.
    # Filter and grid
    par = ACFilterPars(L_fc=.15*base.L)
    ac_filter = model.LFilter(par)
    # par = ACFilterPars(
    #     L_fc=.081*base.L, L_fg=.073*base.L, C_f=.035*base.C, u_fs0=base.u)
    # ac_filter = model.LCLFilter(par)
    ac_source = model.SignalInjection(w_g=base.w, abs_e_g=base.u)
    # Inverter with constant DC voltage
    converter = model.VoltageSourceConverter(u_dc=650)

    # Create system model
    mdl = model.GridConverterIdentification(
        converter, ac_filter, ac_source, delay=identification_cfg.delay)

    # Configure the control system.
    cfg = control.GridFollowingControlCfg(
        L=.15*base.L,
        nom_u=base.u,
        nom_w=base.w,
        max_i=1.5*base.i,
        T_s=1/10e3,
        k_comp=identification_cfg.k_comp)
    ctrl = control.GridFollowingControl(cfg)

    return identification_cfg, mdl, ctrl


# %%
@dataclass
class AdmittanceIdentificationCfg:
    """
    Configuration parameters for converter output admittance identification.

    Parameters
    ----------
    op_point : SimpleNamespace
        SimpleNamespace object containing the operating point values
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
    freqs : ndarray, optional
        Manually specified array of frequencies (Hz) to measure admittance at.
        The default is None, and then f_start, f_stop and n_freqs are used.
    t0 : float, optional
        Stop time for initial simulating to the operating point (s). Should be
        large enough to reach steady-state. The default is 0.1.
    t1 : float, optional
        Additional simulation time for reaching steady-state during signal
        injection (s). The default is 0.02.
    T_eval : float, optional
        Sampling period for the solver. The default is 1e-5.
    n_periods : int, optional
        Number of excitation signal periods to use for calculating the DFT. The
        default is 10.
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
        is zero.
    k_comp : float, optional
        Compensation factor for the delay effect on the converter output
        voltage vector angle. The default is 1.5.

    """

    op_point: SimpleNamespace
    abs_u_e: float
    f_start: float
    f_stop: float
    n_freqs: int
    spacing: str = "log"
    freqs: np.ndarray = None
    t0: float = .1
    t1: float = .02
    T_eval: float = 1e-5
    n_periods: int = 10
    multiprocess: bool = True
    plot_style: str = "re_im"
    filename: str = None
    delay: int = 0
    k_comp: float = 1.5

    def __post_init__(self):
        if self.freqs is None:
            if self.spacing == "lin":
                self.freqs = np.linspace(
                    self.f_start, self.f_stop, self.n_freqs)
            else:
                self.freqs = np.geomspace(
                    self.f_start, self.f_stop, self.n_freqs)
        # Increase excitation signal amplitude with the frequency
        self.amplitudes = self.abs_u_e*np.linspace(1., 5., np.size(self.freqs))


# %%


def save_mat(data, filename):
    """Save the identification results in a .mat-file."""

    # Convert the SimpleNamespace object to dict
    data_dict = dict(data.__dict__.items())
    # Create the file path
    timestamp = datetime.now().strftime("%Y%m%d_%H.%M_")
    filepath = join("matfiles", timestamp + filename + ".mat")

    try:
        savemat(filepath, data_dict)
        print(f"Data successfully exported to {timestamp + filename}")
    except Exception as error:
        print(f"Error saving data: {str(error)}")


def dft(cfg, u, f_e):
    """
    Single-frequency discrete Fourier transform.

    Calculates the frequency component y at frequency f_e from input
    signal u, using the discrete Fourier transform algorithm.
    
    """

    n = int(cfg.n_periods/(f_e*cfg.T_eval))
    u = u[-n:]
    y = 2/n*np.sum(u*np.exp(-2j*np.pi*f_e*cfg.T_eval*np.arange(n)))
    return y


def copy_state(cfg, sim):
    """Make a copy of the simulation state."""
    # Copy the subsystems
    ac_filter = copy.deepcopy(sim.mdl.ac_filter)
    converter = copy.deepcopy(sim.mdl.converter)
    ac_source = copy.deepcopy(sim.mdl.ac_source)
    converter.sol_q_cs = []
    mdl = model.GridConverterIdentification(
        converter, ac_filter, ac_source, delay=cfg.delay)
    for subsystem in mdl.subsystems:
        if hasattr(subsystem, "sol_states"):
            for attr in vars(subsystem.sol_states):
                subsystem.sol_states.__dict__[attr] = []
    mdl.sol_t = []
    mdl.t0 = sim.mdl.sol_t[-1]
    ctrl = copy.deepcopy(sim.ctrl)
    return mdl, ctrl


def pre_process(cfg, mdl, ctrl):
    """Simulate the system to the desired operating point."""

    # Set appropriate references
    ctrl.ref.p_g = cfg.op_point.p_g
    if hasattr(cfg.op_point, "q_g"):
        ctrl.ref.q_g = cfg.op_point.q_g
    if hasattr(cfg.op_point, "v_c"):
        ctrl.ref.v_c = cfg.op_point.v_c
    # Create Simulation object and simulate
    sim = model.Simulation(mdl, ctrl)
    sim.simulate(t_stop=cfg.t0)
    return sim


def identify(cfg, sim_op, i, f_e):
    """Calculate the output admittance at a single frequency."""

    # 1) d-axis injection
    mdl, ctrl = copy_state(cfg, sim_op)
    mdl.ac_source.par.f_e = f_e
    mdl.ac_source.par.abs_u_ed = cfg.amplitudes[i]
    # Set new stop time and simulate
    t_stop = cfg.t0 + cfg.t1 + cfg.n_periods/f_e
    sim = model.Simulation(mdl, ctrl)
    sim.simulate(t_stop=t_stop, T_eval=cfg.T_eval)

    # Transform the voltage and current to synchronous coordinates and
    # calculate the DFT
    u_g1 = np.conj(
        sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.u_gs
    u_gd1 = dft(cfg, u_g1.real, f_e)
    u_gq1 = dft(cfg, u_g1.imag, f_e)
    i_g1 = np.conj(
        sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.i_gs
    i_gd1 = dft(cfg, i_g1.real, f_e)
    i_gq1 = dft(cfg, i_g1.imag, f_e)

    # 2) q-axis injection
    mdl, ctrl = copy_state(cfg, sim_op)
    mdl.ac_source.par.f_e = f_e
    mdl.ac_source.par.abs_u_eq = cfg.amplitudes[i]
    sim = model.Simulation(mdl, ctrl)
    sim.simulate(t_stop=t_stop, T_eval=cfg.T_eval)

    # DFT
    u_g2 = np.conj(
        sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.u_gs
    u_gd2 = dft(cfg, u_g2.real, f_e)
    u_gq2 = dft(cfg, u_g2.imag, f_e)
    i_g2 = np.conj(
        sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.i_gs
    i_gd2 = dft(cfg, i_g2.real, f_e)
    i_gq2 = dft(cfg, i_g2.imag, f_e)

    # Calculate the elements of the output admittance matrix
    I = np.array([[i_gd1, i_gd2], [i_gq1, i_gq2]])
    U = np.array([[u_gd1, u_gd2], [u_gq1, u_gq2]])
    inv_U = np.linalg.inv(U)
    Y_c = -1*I @ inv_U

    Y_dd = Y_c[0, 0]
    Y_qd = Y_c[0, 1]
    Y_dq = Y_c[1, 0]
    Y_qq = Y_c[1, 1]
    return [i, f_e, Y_dd, Y_qd, Y_dq, Y_qq]


def post_process(results):
    """Transform the results to ndarray format."""
    results = np.vstack(results)
    data1 = SimpleNamespace()
    data1.f_e = np.real(results[:, 1])
    data1.Y_dd = results[:, 2]
    data1.Y_qd = results[:, 3]
    data1.Y_dq = results[:, 4]
    data1.Y_qq = results[:, 5]
    return data1


def run_identification():
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
                    callback=collect_result)
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
    if cfg.plot_style is not None:
        plot_identification(data, cfg.plot_style)


if __name__ == '__main__':
    # Run the identification
    run_identification()
