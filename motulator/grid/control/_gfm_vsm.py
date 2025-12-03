"""Virtual synchronous machine control for grid converters."""

from cmath import exp
from dataclasses import dataclass
from math import pi

from motulator.common.control._base import TimeSeries
from motulator.common.control._controllers import ComplexPIController, PIController
from motulator.common.utils._utils import wrap
from motulator.grid.control._base import Measurements
from motulator.grid.control._common import CurrentLimiter
from motulator.grid.control._gfl_current_vector import PLL


# %%
@dataclass
class References:
    """Reference signals for VSM control."""

    T_s: float = 0.0
    u_c: complex = 0j
    i_c: complex = 0j
    v_c: float = 0.0
    u_g: complex = 0j
    p_g: float = 0.0
    q_g: float = 0.0
    d_w_c: float = 0.0
    w_c: float = 0.0
    u_dc: float | None = None


@dataclass
class Feedbacks:
    """Feedback signals for the control system."""

    i_c: complex = 0j
    u_c: complex = 0j
    u_g: complex = 0j
    u_g_flt: complex = 0j
    p_g: float = 0.0
    q_g: float = 0.0
    theta_c: float = 0.0  # Angle of the coordinate system (rad)
    w_c: float = 0.0  # Angular speed of the coordinate system (rad/s)
    w_g: float = 0.0


class VirtualSynchronousMachineController:
    """
    Virtual synchronous machine controller.

    Parameters
    ----------
    u_nom : float
        Nominal grid voltage (V), line-to-neutral peak value.
    w_nom : float
        Nominal grid angular frequency (rad/s).
    i_max : float
        Maximum current (A), peak value.
    R : float, optional
        Total series resistance (Ω), defaults to 0.
    H : float, optional
        Inertia constant (s), defaults to 5.0.
    sigma : float, optional
        Active power droop (p.u.), defaults to 0.05.
    k_D : float, optional
        Virtual mechanical damping constant (p.u.), defaults to 50.0.
    alpha_f : float, optional
        Damping low-pass filter bandwidth (rad/s), defaults to 2*pi*1.
    w_b : float, optional
        Current low-pass filter bandwidth (rad/s), defaults to 2*pi*5.
    T_s : float, optional
        Sampling period (s), defaults to 125e-6.

    References
    ----------
    .. [#Har2020] Harnefors, Rahman, Hinkkanen, Routimo, "Reference-feedforward
       power-synchronization control," IEEE Trans. Power Electron., 2020,
       https://doi.org/10.1109/TPEL.2020.2970991

    """

    def __init__(
        self,
        u_nom: float,
        w_nom: float,
        i_nom: float,
        i_max: float,
        L: float,
        alpha_c: float = 2 * pi * 400,
        alpha_v: float = 2 * pi * 25,
        R: float = 0.0,
        H: float = 1.0,
        sigma: float = 0.05,
        k_D: float = 0.0,
        k_q: float = 0.3,
        k_pu: float = 0.3,
        k_iu: float = 180.0,
        # k_pi: float = 3.0,
        # k_ii: float = 100.0,
        w_b: float = 1 / 5e-3,
        alpha_f: float = 2 * pi * 1,
        alpha_pll: float = 2 * pi * 20,
        T_s: float = 125e-6,
    ) -> None:
        self.u_g_flt: complex = u_nom
        self.theta_c: float = 0.0
        self.w_c: float = w_nom
        self.w_f: float = w_nom
        self.d_voltage_ctrl = PIController(k_p=k_pu, k_i=k_iu)
        self.q_voltage_ctrl = PIController(k_p=k_pu, k_i=k_iu)
        # self.d_current_ctrl = PIController(k_p=k_pi, k_i=k_ii)
        # self.q_current_ctrl = PIController(k_p=k_pi, k_i=k_ii)
        # self.voltage_ctrl = ComplexPIController(
        #     k_t=alpha_v * L, k_i=alpha_v * alpha_v * L, k_p=2 * alpha_v * L
        # )
        self.current_ctrl = ComplexPIController(
            k_t=alpha_c * L, k_i=alpha_c * alpha_c * L, k_p=2 * alpha_c * L
        )
        self.current_limiter = CurrentLimiter(i_max)
        self.pll = PLL(u_nom, w_nom, alpha_pll)
        self.u_nom = u_nom
        self.w_nom = w_nom
        self.i_nom = i_nom
        self.s_base = 1.5 * u_nom * i_nom
        self.k_g = self.s_base / (sigma * w_nom)
        self.R = R
        self.H = H
        self.k_D = k_D
        self.k_q = k_q
        self.w_b = w_b
        self.alpha_f = alpha_f
        self.M = 2 * self.s_base * H / w_nom
        self.T_s = T_s

    def get_feedback(self, u_c_ab: complex, meas: Measurements) -> Feedbacks:
        """Get the feedback signals."""
        out = Feedbacks(w_c=self.w_c, theta_c=self.theta_c)
        self.pll_outputs = self.pll.compute_output(u_c_ab, meas.i_c_ab, meas.u_g_ab)

        # Transform the measured values into synchronous coordinates
        out.i_c = exp(-1j * out.theta_c) * meas.i_c_ab
        out.u_c = exp(-1j * out.theta_c) * u_c_ab
        out.u_g = exp(-1j * out.theta_c) * meas.u_g_ab

        # Other feedback signals
        p_loss = 1.5 * self.R * abs(out.i_c) ** 2
        out.p_g = 1.5 * (out.u_c * out.i_c.conjugate()).real - p_loss
        out.q_g = 1.5 * (out.u_c * out.i_c.conjugate()).imag
        out.w_g = self.pll_outputs.w_g
        out.u_g_flt = self.u_g_flt
        return out

    def compute_output(
        self, p_g_ref: float, v_c_ref: float, fbk: Feedbacks
    ) -> References:
        """Compute references."""
        ref = References(T_s=self.T_s, v_c=v_c_ref, p_g=p_g_ref, w_c=self.w_c)

        # Active power control
        # p_gov = (
        #     ref.p_g / self.s_base
        #     + (1 - fbk.w_c / self.w_nom) * self.k_g * self.w_nom / self.s_base
        # )
        # p_d = self.k_D * (fbk.w_c - self.w_f) / self.w_nom
        # ref.d_w_c = self.w_nom * (p_gov - p_d - fbk.p_g / self.s_base) / self.H
        p_gov = ref.p_g + self.k_g * (self.w_nom - fbk.w_c)
        p_d = self.k_D * (fbk.w_c - self.w_f)
        ref.d_w_c = (p_gov - fbk.p_g - p_d) / self.M

        # Reactive power-voltage droop
        ref.u_g = ref.v_c * (
            ref.v_c / self.u_nom + self.k_q * (ref.q_g - fbk.q_g) / self.s_base
        )

        # Current reference
        i_cd = self.d_voltage_ctrl.compute_output(ref.u_g.real, self.u_g_flt.real)
        i_cq = self.q_voltage_ctrl.compute_output(0.0, self.u_g_flt.imag)
        ref.i_c = self.current_limiter(i_cd + 1j * i_cq)

        # # Voltage reference
        # u_cd = self.d_current_ctrl.compute_output(
        #     ref.i_c.real, fbk.i_c.real / self.i_nom
        # )
        # u_cq = self.q_current_ctrl.compute_output(
        #     ref.i_c.imag, fbk.i_c.imag / self.i_nom
        # )
        # ref.u_c = (
        #     u_cd
        #     + 1.0049 * (ref.u_g.real + fbk.i_c.imag / self.i_nom * 0.00047746 * fbk.w_c)
        #     + 1j * (u_cq - 1.0049 * fbk.i_c.real / self.i_nom * 0.00047746 * fbk.w_c)
        # ) * self.u_nom
        # ref.u_g = ref.u_g * self.u_nom
        # ref.i_c = ref.i_c * self.i_nom
        # ref.i_c = self.voltage_ctrl.compute_output(ref.u_g, fbk.u_g_flt)
        # ref.i_c = self.current_limiter(ref.i_c)
        ref.u_c = self.current_ctrl.compute_output(ref.i_c, fbk.i_c, ref.u_g)
        return ref

    def update(self, ref: References, fbk: Feedbacks) -> None:
        """Update states."""
        self.u_g_flt += ref.T_s * self.w_b * (fbk.u_g - self.u_g_flt)
        self.w_f += ref.T_s * self.alpha_f * (fbk.w_c - self.w_f)
        self.w_c += ref.T_s * ref.d_w_c
        self.theta_c += ref.T_s * ref.w_c
        self.theta_c = wrap(self.theta_c)
        self.d_voltage_ctrl.update(ref.T_s, fbk.i_c.real)
        self.q_voltage_ctrl.update(ref.T_s, fbk.i_c.imag)
        # self.d_current_ctrl.update(ref.T_s, fbk.u_c.real / self.u_nom)
        # self.q_current_ctrl.update(ref.T_s, fbk.u_c.imag / self.u_nom)
        # self.voltage_ctrl.update(ref.T_s, fbk.i_c, fbk.w_c)
        self.current_ctrl.update(ref.T_s, fbk.u_c, fbk.w_c)
        self.pll.update(ref.T_s, self.pll_outputs)

    def post_process(self, ts: TimeSeries) -> None:
        """Post-process controller time series."""
