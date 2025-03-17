"""Example plotting scripts for grid converters."""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from motulator.common.utils import complex2abc
from motulator.grid import model

# Plotting parameters
plt.rcParams["axes.prop_cycle"] = cycler(color="brgcmyk")
plt.rcParams["lines.linewidth"] = 1.
plt.rcParams["axes.grid"] = True
plt.rcParams.update({"text.usetex": False})


# %%
def plot(sim, base=None, plot_pcc_voltage=True, plot_w=False, t_span=None):
    """
    Plot example figures of grid converter simulations.

    Parameters
    ----------
    sim : Simulation
        Should contain the simulated data.
    base : BaseValues, optional
        Base values for scaling the waveforms. If not given, plots the figures
        in SI units.
    plot_pcc_voltage : bool, optional
        If True, the phase voltage waveforms are plotted at the point of common
        coupling (PCC). Otherwise, the grid voltage waveforms are plotted. The
        default is True.
    plot_w : bool, optional
        If True, plot the grid frequency. Otherwise, plot the phase angle. The
        default is False.
    t_span : 2-tuple, optional
        Time span. The default is (0, sim.ctrl.ref.t[-1]).

    """

    mdl = sim.mdl  # Continuous-time data
    ctrl = sim.ctrl.data  # Discrete-time data
    ctrl.t = ctrl.ref.t  # Discrete time

    # Check if the time span was given
    if t_span is None:
        t_span = (0, ctrl.t[-1])  # Time span

    # Check if the base values were given
    if base is None:
        pu_vals = False
        # Scaling with unity base values except for power use kW
        base = SimpleNamespace(w=1, u=1, i=1, p=1000)
    else:
        pu_vals = True

    # Three-phase quantities
    i_g_abc = complex2abc(mdl.ac_filter.data.i_gs).T
    e_g_abc = complex2abc(mdl.ac_source.data.e_gs).T
    u_g_abc = complex2abc(mdl.ac_filter.data.u_gs).T

    # Calculation of active and reactive powers
    p_g = 1.5*np.real(mdl.ac_filter.data.e_gs*np.conj(mdl.ac_filter.data.i_gs))
    q_g = 1.5*np.imag(mdl.ac_filter.data.e_gs*np.conj(mdl.ac_filter.data.i_gs))

    # Coordinate transformation in the case of observer-based GFM control
    if hasattr(sim.ctrl, "observer"):
        # Convert quantities to converter-output-voltage coordinates
        T = np.where(
            np.abs(ctrl.fbk.v_c) > 0,
            np.conj(ctrl.fbk.v_c)/np.abs(ctrl.fbk.v_c), 1)
        ctrl.ref.u_c = T*ctrl.ref.u_c
        ctrl.fbk.i_c = T*ctrl.fbk.i_c
        ctrl.ref.i_c = T*ctrl.ref.i_c

    # %%
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))

    if mdl.converter.par.C_dc is None:
        if not plot_pcc_voltage:
            # Subplot 1: Grid voltage
            ax1.plot(
                mdl.ac_source.data.t,
                e_g_abc/base.u,
                label=[
                    r"$e_\mathrm{ga}$", r"$e_\mathrm{gb}$", r"$e_\mathrm{gc}$"
                ])
        else:
            # Subplot 1: PCC voltage
            ax1.plot(
                mdl.ac_filter.data.t,
                u_g_abc/base.u,
                label=[
                    r"$u_\mathrm{ga}$", r"$u_\mathrm{gb}$", r"$u_\mathrm{gc}$"
                ])
    else:
        # Subplot 1: DC-bus voltage
        ax1.plot(
            mdl.converter.data.t,
            mdl.converter.data.u_dc/base.u,
            label=r"$u_\mathrm{dc}$")
        ax1.plot(
            ctrl.t,
            ctrl.ref.u_dc/base.u,
            "--",
            label=r"$u_\mathrm{dc,ref}$",
            ds="steps-post")
    ax1.legend()
    ax1.set_xlim(t_span)
    ax1.set_xticklabels([])

    # Subplot 2: Grid currents
    ax2.plot(
        mdl.ac_filter.data.t,
        i_g_abc/base.i,
        label=[r"$i_\mathrm{ga}$", r"$i_\mathrm{gb}$", r"$i_\mathrm{gc}$"])
    ax2.legend()
    ax2.set_xlim(t_span)
    ax2.set_xticklabels([])

    if plot_w:
        # Subplot 3: Grid and converter frequencies
        ax3.plot(
            mdl.ac_source.data.t,
            mdl.ac_source.data.w_g/base.w,
            label=r"$\omega_\mathrm{g}$")
        ax3.plot(
            ctrl.t,
            ctrl.fbk.w_c/base.w,
            "--",
            label=r"$\omega_\mathrm{c}$",
            ds="steps-post")
        ax3.legend()
        ax3.set_xlim(t_span)
    else:
        # Subplot 3: Phase angles
        ax3.plot(
            mdl.ac_source.data.t,
            mdl.ac_source.data.theta_g,
            label=r"$\theta_\mathrm{g}$")
        ax3.plot(
            ctrl.t,
            ctrl.fbk.theta_c,
            "--",
            label=r"$\theta_\mathrm{c}$",
            ds="steps-post")
    ax3.legend()
    ax3.set_xlim(t_span)

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel("Voltage (p.u.)")
        ax2.set_ylabel("Current (p.u.)")
    else:
        ax1.set_ylabel("Voltage (V)")
        ax2.set_ylabel("Current (A)")
    if not plot_w:
        ax3.set_ylabel("Angle (rad)")
    elif pu_vals:
        ax3.set_ylabel("Frequency (p.u.)")
    else:
        ax3.set_ylabel("Frequency (rad/s)")
    ax3.set_xlabel("Time (s)")

    fig.align_ylabels()
    plt.tight_layout()
    plt.grid()
    ax3.grid()
    #plt.show()

    # %%
    # Second figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))

    # Subplot 1: Active and reactive power
    ax1.plot(mdl.ac_filter.data.t, p_g/base.p, label=r"$p_\mathrm{g}$")
    ax1.plot(mdl.ac_filter.data.t, q_g/base.p, label=r"$q_\mathrm{g}$")
    ax1.plot(
        ctrl.t,
        ctrl.ref.p_g/base.p,
        "--",
        label=r"$p_\mathrm{g,ref}$",
        ds="steps-post")
    if hasattr(ctrl.ref, "q_g"):
        ax1.plot(
            ctrl.t,
            ctrl.ref.q_g/base.p,
            "--",
            label=r"$q_\mathrm{g,ref}$",
            ds="steps-post")
    ax1.legend()
    ax1.set_xlim(t_span)
    ax1.set_xticklabels([])

    # Subplot 2: Converter currents
    ax2.plot(
        ctrl.t,
        np.real(ctrl.fbk.i_c/base.i),
        label=r"$i_\mathrm{cd}$",
        ds="steps-post")
    ax2.plot(
        ctrl.t,
        np.imag(ctrl.fbk.i_c/base.i),
        label=r"$i_\mathrm{cq}$",
        ds="steps-post")
    ax2.plot(
        ctrl.t,
        np.real(ctrl.ref.i_c/base.i),
        "--",
        label=r"$i_\mathrm{cd,ref}$",
        ds="steps-post")
    ax2.plot(
        ctrl.t,
        np.imag(ctrl.ref.i_c/base.i),
        "--",
        label=r"$i_\mathrm{cq,ref}$",
        ds="steps-post")
    ax2.legend()
    ax2.set_xlim(t_span)
    ax2.set_xticklabels([])

    if hasattr(sim.ctrl, "observer"):
        # Subplot 3: Converter voltage reference, quasi-static converter
        # voltage, and grid voltage magnitudes
        ax3.plot(
            ctrl.t,
            np.abs(ctrl.ref.u_c/base.u),
            label=r"$u_\mathrm{c,ref}$",
            ds="steps-post")
        ax3.plot(
            ctrl.t,
            np.abs(ctrl.fbk.v_c/base.u),
            label=r"$\hat{v}_\mathrm{c}$",
            ds="steps-post")
        ax3.plot(
            mdl.ac_source.data.t,
            np.abs(mdl.ac_source.data.e_gs/base.u),
            "k--",
            label=r"$e_\mathrm{g}$")

    elif isinstance(mdl, model.GridConverterIdentification):
        # Subplot 3: Synchronous frame grid voltage
        e_g = np.conj(mdl.ac_source.data.exp_j_theta_g)*mdl.ac_source.data.e_gs

        ax3.plot(
            mdl.ac_source.data.t,
            np.real(e_g/base.u),
            label=r"$e_\mathrm{gd}$",
            ds="steps-post")
        ax3.plot(
            mdl.ac_source.data.t,
            np.imag(e_g/base.u),
            label=r"$e_\mathrm{gq}$",
            ds="steps-post")

    else:
        # Subplot 3: Converter voltage reference and grid voltage magnitude
        ax3.plot(
            ctrl.t,
            np.real(ctrl.fbk.u_c/base.u),
            label=r"$u_\mathrm{cd}$",
            ds="steps-post")
        ax3.plot(
            ctrl.t,
            np.imag(ctrl.fbk.u_c/base.u),
            label=r"$u_\mathrm{cq}$",
            ds="steps-post")
        ax3.plot(
            mdl.ac_source.data.t,
            np.abs(mdl.ac_source.data.e_gs)/base.u,
            "k--",
            label=r"$e_\mathrm{g}$")

    ax3.legend()
    ax3.set_xlim(t_span)

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel("Power (p.u.)")
        ax2.set_ylabel("Current (p.u.)")
        ax3.set_ylabel("Voltage (p.u.)")
    else:
        ax1.set_ylabel("Power (kW, kVar)")
        ax2.set_ylabel("Current (A)")
        ax3.set_ylabel("Voltage (V)")
    ax3.set_xlabel("Time (s)")

    fig.align_ylabels()
    plt.tight_layout()
    plt.grid()
    ax3.grid()
    plt.show()


# %%
def plot_voltage_vector(sim, base=None):
    """
    Plot locus of the grid voltage vector.

    Parameters
    ----------
    sim : Simulation
        Should contain the simulated data.
    base : BaseValues, optional
        Base values for scaling the waveforms.

    """

    mdl = sim.mdl  # Continuous-time data

    # Check if the base values were given
    if base is None:
        pu_vals = False
        # Scaling with unity base values except for power use kW
        base = SimpleNamespace(w=1, u=1, i=1, p=1000)
    else:
        pu_vals = True

    # Plot the grid voltage vector in the complex plane
    _, ax = plt.subplots()
    ax.plot(
        mdl.ac_source.data.e_gs.real/base.u,
        mdl.ac_source.data.e_gs.imag/base.u,
        label="Grid voltage")
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    if pu_vals:
        ax.set_xlabel("Real (p.u.)")
        ax.set_ylabel("Imaginary (p.u.)")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        ax.set_xlabel("Real (V)")
        ax.set_ylabel("Imaginary (V)")
    ax.legend()
    ax.set_aspect("equal")


# %%
def plot_identification(data_sim, plot_style):
    """
    Plot elements of the identified converter output admittance matrix

    Parameters
    ----------
    data_sim : SimpleNamespace
        Contains the elements of the output admittance matrix and corresponding
        frequencies.
    plot_style : str
        Set this variable to plot either the real and imaginary parts of the
        admittance ("re_im") or the magnitude and phase ("bode").

    """
    # plt.rcParams.update({"text.usetex": True})

    fig, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(
        4, 2, figsize=(9, 6))
    if plot_style == "bode":
        ylim = (-0.01, 0.11)
        ylim_angle = (-190, 190)
        yticks = ([0, 0.05, 0.1])
        yticks_angle = ([-180, -90, 0, 90, 180])
        ylabels_angle = (["", "-90", "", "90", ""])

        ax1.semilogx(data_sim.f_e, np.abs(data_sim.Y_dd))
        ax2.semilogx(data_sim.f_e, np.angle(data_sim.Y_dd, deg=True))
        ax3.semilogx(data_sim.f_e, np.abs(data_sim.Y_dq))
        ax4.semilogx(data_sim.f_e, np.angle(data_sim.Y_dq, deg=True))
        ax5.semilogx(data_sim.f_e, np.abs(data_sim.Y_qd))
        ax6.semilogx(data_sim.f_e, np.angle(data_sim.Y_qd, deg=True))
        ax7.semilogx(data_sim.f_e, np.abs(data_sim.Y_qq))
        ax8.semilogx(data_sim.f_e, np.angle(data_sim.Y_qq, deg=True))

        ax1.set_ylabel(r"$|Y_\mathrm{dd}|\ (\Omega^{-1})$")
        ax2.set_ylabel(r"$\angle Y_\mathrm{dd}\ (\mathrm{deg})$")
        ax3.set_ylabel(r"$|Y_\mathrm{dq}|\ (\Omega^{-1})$")
        ax4.set_ylabel(r"$\angle Y_\mathrm{dq}\ (\mathrm{deg})$")
        ax5.set_ylabel(r"$|Y_\mathrm{qd}|\ (\Omega^{-1})$")
        ax6.set_ylabel(r"$\angle Y_\mathrm{qd}\ (\mathrm{deg})$")
        ax7.set_ylabel(r"$|Y_\mathrm{qq}|\ (\Omega^{-1})$")
        ax8.set_ylabel(r"$\angle Y_\mathrm{qq}\ (\mathrm{deg})$")

        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim_angle)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim_angle)
        ax5.set_ylim(ylim)
        ax6.set_ylim(ylim_angle)
        ax7.set_ylim(ylim)
        ax8.set_ylim(ylim_angle)

        ax1.set_yticks(yticks)
        ax2.set_yticks(yticks_angle)
        ax2.set_yticklabels(ylabels_angle)
        ax3.set_yticks(yticks)
        ax4.set_yticks(yticks_angle)
        ax4.set_yticklabels(ylabels_angle)
        ax5.set_yticks(yticks)
        ax6.set_yticks(yticks_angle)
        ax6.set_yticklabels(ylabels_angle)
        ax7.set_yticks(yticks)
        ax8.set_yticks(yticks_angle)
        ax8.set_yticklabels(ylabels_angle)

    else:
        ylim = (-0.05, 0.11)

        ax1.semilogx(data_sim.f_e, np.real(data_sim.Y_dd))
        ax2.semilogx(data_sim.f_e, np.imag(data_sim.Y_dd))
        ax3.semilogx(data_sim.f_e, np.real(data_sim.Y_dq))
        ax4.semilogx(data_sim.f_e, np.imag(data_sim.Y_dq))
        ax5.semilogx(data_sim.f_e, np.real(data_sim.Y_qd))
        ax6.semilogx(data_sim.f_e, np.imag(data_sim.Y_qd))
        ax7.semilogx(data_sim.f_e, np.real(data_sim.Y_qq))
        ax8.semilogx(data_sim.f_e, np.imag(data_sim.Y_qq))

        ax1.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{dd}\}\ (\Omega^{-1})$")
        ax2.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{dd}\}\ (\Omega^{-1})$")
        ax3.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{dq}\}\ (\Omega^{-1})$")
        ax4.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{dq}\}\ (\Omega^{-1})$")
        ax5.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{qd}\}\ (\Omega^{-1})$")
        ax6.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{qd}\}\ (\Omega^{-1})$")
        ax7.set_ylabel(r"$\mathrm{Re}\{Y_\mathrm{qq}\}\ (\Omega^{-1})$")
        ax8.set_ylabel(r"$\mathrm{Im}\{Y_\mathrm{qq}\}\ (\Omega^{-1})$")

        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim)
        ax5.set_ylim(ylim)
        ax6.set_ylim(ylim)
        ax7.set_ylim(ylim)
        ax8.set_ylim(ylim)

    ax4.set_xlabel(r"Frequency (Hz)")
    ax8.set_xlabel(r"Frequency (Hz)")

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])
    ax7.set_xticklabels([])

    fig.align_ylabels()
    plt.show()


# %%
def save_plot(name):
    """
    Save figures.

    This saves figures in a folder "figures" in the current directory. If the
    folder does not exist, it is created.

    Parameters
    ----------
    name : string
        Name for the figure

    """
    plt.savefig(name + ".pdf")
