"""
Three-phase voltage source model.

Peak-valued complex space vectors are used.

"""
from types import SimpleNamespace

import numpy as np

from motulator.common.model import Subsystem


# %%
class ThreePhaseVoltageSource(Subsystem):
    """
    Three-phase voltage source model.

    The frequency, phase shift, and magnitude can be given either as constants
    or functions of time. An unbalanced source can be modeled by specifying a
    negative-sequence component. Notice that the zero-sequence component is not
    included in this model.

    Parameters
    ----------
    w_g : float | callable
        Angular frequency (rad/s).
    abs_e_g : float | callable
        Magnitude of the positive-sequence component (peak value).
    phi : float | callable, optional
        Phase shift (rad) of the positive-sequence component. The default is 0.
    abs_e_g_neg : float | callable, optional
        Magnitude of the negative-sequence component (peak value). The default
        is 0.
    phi_neg : float | callable, optional
        Phase shift (rad) of the negative-sequence component. The default is 0.

    Notes
    -----
    This model is typically used to represent a voltage source, but it can be
    configured to represent, e.g., a current source as well.

    """

    def __init__(self, w_g, abs_e_g, phi=0, abs_e_g_neg=0, phi_neg=0):
        super().__init__()
        self.par = SimpleNamespace(
            w_g=w_g,
            abs_e_g=abs_e_g,
            phi=phi,
            abs_e_g_neg=abs_e_g_neg,
            phi_neg=phi_neg)
        self.state = SimpleNamespace(exp_j_theta_g=complex(1))
        self.sol_states = SimpleNamespace(exp_j_theta_g=[])

    def _get_value(self, t, par):
        return par(t) if callable(par) else par

    def generate_space_vector(self, t, exp_j_theta_g):
        """Generate the space vector in stationary coordinates."""
        abs_e_g = self._get_value(t, self.par.abs_e_g)
        phi = self._get_value(t, self.par.phi)
        abs_e_g_neg = self._get_value(t, self.par.abs_e_g_neg)
        phi_neg = self._get_value(t, self.par.phi_neg)

        # Space vector in stationary coordinates
        e_gs = abs_e_g*exp_j_theta_g*np.exp(1j*phi)
        # Add possible negative sequence component
        e_gs += abs_e_g_neg*np.conj(exp_j_theta_g*np.exp(1j*phi_neg))

        return e_gs

    def set_outputs(self, t):
        """Set output variables."""
        self.out.e_gs = self.generate_space_vector(t, self.state.exp_j_theta_g)

    def set_inputs(self, t):
        """Set input variables."""
        self.inp.w_g = self._get_value(t, self.par.w_g)

    def rhs(self):
        """Compute the state derivative."""
        d_exp_j_theta_g = 1j*self.inp.w_g*self.state.exp_j_theta_g
        return [d_exp_j_theta_g]

    def post_process_states(self):
        """Post-process the solution."""
        self.data.w_g = np.vectorize(self._get_value)(
            self.data.t, self.par.w_g)
        self.data.theta_g = np.angle(self.data.exp_j_theta_g)
        self.data.e_gs = self.generate_space_vector(
            self.data.t, self.data.exp_j_theta_g)


# %%
class SignalInjection(Subsystem):
    """
    Three-phase voltage source model with small-signal excitation.

    The frequency, phase shift, and magnitude can be given either as constants
    or functions of time. Notice that the zero-sequence component is not
    included in this model.

    Parameters
    ----------
    w_g : float | callable
        Angular frequency (rad/s).
    abs_e_g : float | callable
        Magnitude of the positive-sequence component (peak value).
    phi : float | callable, optional
        Phase shift (rad) of the positive-sequence component. The default is 0.

    Notes
    -----
    This model is typically used to represent a voltage source, but it can be
    configured to represent, e.g., a current source as well.

    """

    def __init__(self, w_g, abs_e_g, phi=0, abs_u_ed=0, abs_u_eq=0, f_e=0):
        super().__init__()
        self.par = SimpleNamespace(
            w_g=w_g,
            abs_e_g=abs_e_g,
            phi=phi,
            abs_u_ed=abs_u_ed,
            abs_u_eq=abs_u_eq,
            f_e=f_e)
        self.state = SimpleNamespace(
            exp_j_theta_g=complex(1), exp_j_theta_e=complex(1))
        self.sol_states = SimpleNamespace(exp_j_theta_g=[], exp_j_theta_e=[])

    def _get_value(self, t, par):
        return par(t) if callable(par) else par

    def generate_space_vector(self, t, exp_j_theta_g, exp_j_theta_e):
        """Generate the space vector in stationary coordinates."""
        abs_e_g = self._get_value(t, self.par.abs_e_g)
        phi = self._get_value(t, self.par.phi)

        # Add excitation signal in synchronous coordinates
        e_g = abs_e_g + (self.par.abs_u_ed +
                         1j*self.par.abs_u_eq)*np.real(exp_j_theta_e)

        # Transform to stationary coordinates
        e_gs = e_g*exp_j_theta_g*np.exp(1j*phi)

        return e_gs

    def set_outputs(self, t):
        """Set output variables."""
        self.out.e_gs = self.generate_space_vector(
            t, self.state.exp_j_theta_g, self.state.exp_j_theta_e)

    def set_inputs(self, t):
        """Set input variables."""
        self.inp.w_g = self._get_value(t, self.par.w_g)

    def rhs(self):
        """Compute the state derivative."""
        d_exp_j_theta_g = 1j*self.inp.w_g*self.state.exp_j_theta_g
        d_exp_j_theta_e = 2*np.pi*1j*self.par.f_e*self.state.exp_j_theta_e
        return [d_exp_j_theta_g, d_exp_j_theta_e]

    def post_process_states(self):
        """Post-process the solution."""
        self.data.w_g = np.vectorize(self._get_value)(
            self.data.t, self.par.w_g)
        self.data.theta_g = np.angle(self.data.exp_j_theta_g)
        self.data.e_gs = self.generate_space_vector(
            self.data.t, self.data.exp_j_theta_g, self.data.exp_j_theta_e)
