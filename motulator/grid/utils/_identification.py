"""Functions and classes for converter output admittance identification."""

import copy
import multiprocessing as mp
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from motulator.grid import model


# %%
@dataclass
class AdmittanceIdentificationCfg:
    """
    Configuration parameters for converter output admittance identification.

    Parameters
    ----------
    op_point : SimpleNamespace
        SimpleNamespace object containing the operating point values for the
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
    freqs : NDArray, optional
        Manually specified array of frequencies (Hz) to measure admittance at.
        The default is None, and then f_start, f_stop and n_freqs are used.
    t0 : float, optional
        Stop time for initial simulating to the operating point (s). Should be set
        large enough to reach steady-state. The default is 0.1.

    """

    op_point: SimpleNamespace
    abs_u_e: float
    f_start: float
    f_stop: float
    n_freqs: int
    spacing: str = "log"
    freqs: np.ndarray = None
    t0: float = .1

    def __post_init__(self):
        if self.freqs is None:
            if self.spacing == "lin":
                self.freqs = np.linspace(
                    self.f_start, self.f_stop, self.n_freqs)
            else:
                self.freqs = np.geomspace(
                    self.f_start, self.f_stop, self.n_freqs)


class AdmittanceIdentification:
    """
    Admittance identification.

    """

    def __init__(self, cfg, mdl, ctrl):
        self.cfg = cfg
        self.mdl = mdl
        self.ctrl = ctrl

    # simulate to op point

    # copy state

    # create multiprocess task

    # FFT

    # plot admittance

    # (optional) plot time domain signals

    def pre_process(self):
        self.ctrl.ref.p_g = lambda t: self.cfg.op_point.p_g
        if hasattr(self.ctrl.ref, "q_g"):
            self.ctrl.ref.q_g = lambda t: self.cfg.op_point.q_g
        if hasattr(self.ctrl.ref, "v_c"):
            self.ctrl.ref.v_c = lambda t: self.cfg.op_point.v_c
        sim = model.Simulation(self.mdl, self.ctrl)
        sim.simulate(t_stop=self.cfg.t0)
        self.mdl, self.ctrl = sim.mdl, sim.ctrl

    def identify(self):
        pass

    def post_process(self):
        pass

    def run(self, multiprocess=True):
        self.pre_process()
