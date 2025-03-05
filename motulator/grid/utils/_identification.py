"""Functions and classes for converter output admittance identification."""

import copy
import multiprocessing as mp
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from motulator.grid import model
from motulator.grid.utils import ACFilterPars, plot


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
    T_eval : float, optional
        Sampling period for the solver. The default is 1e-5.
    n_periods : int, optional
        Number of excitation signal periods to simulate for each frequency. The
        default is 2.

    """

    op_point: SimpleNamespace
    abs_u_e: float
    f_start: float
    f_stop: float
    n_freqs: int
    spacing: str = "log"
    freqs: np.ndarray = None
    t0: float = .1
    T_eval: float = 1e-5
    n_periods: int = 2

    def __post_init__(self):
        if self.freqs is None:
            if self.spacing == "lin":
                self.freqs = np.linspace(
                    self.f_start, self.f_stop, self.n_freqs)
            else:
                self.freqs = np.geomspace(
                    self.f_start, self.f_stop, self.n_freqs)


# %%
class AdmittanceIdentification:
    """
    Admittance identification.

    """

    def __init__(self, cfg, mdl, ctrl):
        self.cfg = cfg
        self.mdl = mdl
        self.ctrl = ctrl
        self.result = []
        self.data = SimpleNamespace()

    # create multiprocess task

    # FFT

    # plot admittance

    # (optional) plot time domain signals

    def collect_result(self, result):
        self.result.append(result)

    def custom_error_callback(self, error):
        print(f'Got error: {error}')

    def dft(self, u, f_e):
        """
        Single-frequency discrete Fourier transform.

        Calculates the frequency component y(k) at frequency k=f_e from input
        signal u, using the discrete Fourier transform algorithm.
        
        """

        y = 0
        return y

    def copy_state(self):
        """Make a copy of the simulation state."""
        par = ACFilterPars(L_fc=self.mdl.ac_filter.par.L_f)
        ac_filter = model.ACFilter(par)
        ac_filter.data = copy.deepcopy(self.mdl.ac_filter.data)
        ac_filter.inp = copy.deepcopy(self.mdl.ac_filter.inp)
        ac_filter.out = copy.deepcopy(self.mdl.ac_filter.out)
        ac_filter.state = copy.deepcopy(self.mdl.ac_filter.state)

        converter = copy.deepcopy(self.mdl.converter)
        converter.sol_q_cs = []
        ac_source = copy.deepcopy(self.mdl.ac_source)
        mdl = model.GridConverterIdentification(
            converter, ac_filter, ac_source, delay=0)
        for subsystem in mdl.subsystems:
            if hasattr(subsystem, "sol_states"):
                for attr in vars(subsystem.sol_states):
                    subsystem.sol_states.__dict__[attr] = []
        mdl.sol_t = []
        mdl.t0 = self.mdl.sol_t[-1]
        ctrl = copy.deepcopy(self.ctrl)
        return mdl, ctrl

    def pre_process(self):
        """Simulate the system to the desired operating point."""
        self.ctrl.ref.p_g = lambda t: self.cfg.op_point.p_g
        if hasattr(self.cfg.op_point, "q_g"):
            self.ctrl.ref.q_g = lambda t: self.cfg.op_point.q_g
        if hasattr(self.cfg.op_point, "v_c"):
            self.ctrl.ref.v_c = lambda t: self.cfg.op_point.v_c
        sim = model.Simulation(self.mdl, self.ctrl)
        sim.simulate(t_stop=self.cfg.t0)
        self.mdl, self.ctrl = sim.mdl, sim.ctrl

    def identify(self, i, f_e):
        """Calculate the output admittance at a single frequency."""

        # 1) d-axis injection
        mdl, ctrl = self.copy_state()
        mdl.ac_source.par.f_e = f_e
        mdl.ac_source.par.abs_u_ed = self.cfg.abs_u_e
        sim = model.Simulation(mdl, ctrl)
        sim.simulate(t_stop=self.cfg.n_periods/f_e, T_eval=self.cfg.T_eval)

        t = sim.mdl.ac_source.data.t
        u_g1 = np.conj(
            sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.u_gs
        u_gd1, u_gq1 = u_g1.real, u_g1.imag
        i_g1 = np.conj(
            sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.i_gs
        i_gd1, i_gq1 = i_g1.real, i_g1.imag

        #plot(sim)

        # 2) q-axis injection
        mdl, ctrl = self.copy_state()
        mdl.ac_source.par.f_e = f_e
        mdl.ac_source.par.abs_u_eq = self.cfg.abs_u_e
        sim = model.Simulation(mdl, ctrl)
        sim.simulate(t_stop=self.cfg.n_periods/f_e, T_eval=self.cfg.T_eval)

        u_g2 = np.conj(
            sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.u_gs
        i_g2 = np.conj(
            sim.mdl.ac_source.data.exp_j_theta_g)*sim.mdl.ac_filter.data.i_gs

        plot(sim)
        Y_dd, Y_qd, Y_dq, Y_qq = 0, 0, 0, 0
        return [i, f_e, Y_dd, Y_qd, Y_dq, Y_qq]

    def post_process(self):
        """Transform the lists to ndarray format."""
        result = np.vstack(self.result)
        self.data.f_e = np.real(result[:, 1])
        self.data.Y_dd = result[:, 2]
        self.data.Y_qd = result[:, 3]
        self.data.Y_dq = result[:, 4]
        self.data.Y_qq = result[:, 5]

    def run(self, multiprocess=True):
        self.pre_process()

        if multiprocess:
            with mp.Pool() as pool:
                for i, f_e in enumerate(self.cfg.freqs):
                    pool.apply_async(
                        self.identify,
                        args=[i, f_e],
                        error_callback=self.custom_error_callback,
                        callback=self.collect_result)
        else:
            for i, f_e in enumerate(self.cfg.freqs):
                result = self.identify(i=i, f_e=f_e)
                self.collect_result(result)

        return self.data
