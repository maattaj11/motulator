"""Script for computing analytical output admittance using GFL control."""

import numpy as np
from sympy import Matrix, exp, eye, lambdify, symbols

from motulator.grid import utils
from motulator.grid.utils._identification import (
    IdentificationResults,
    plot_identification,
    plot_vector_diagram,
    save_csv,  # noqa: F401
)

I = eye(2)  # noqa: E741
J = Matrix([[0, -1], [1, 0]])
s, omega = symbols("s omega")

# Calculate base values
nom = utils.NominalValues(U=400, I=14.5, f=50, P=10e3)
base = utils.BaseValues.from_nominal(nom)

# Model parameters
e_g0mag = base.u
omega_g = base.w
L_f = 0.2 * base.L

# Define the operating-point grid active and reactive powers
p_g0 = 0.5 * base.p
q_g0 = 0.5 * base.p

# Calculate other operating-point quantities, L_g=0 is assumed
u_gmag0 = base.u
i_gd0 = 2 * p_g0 / (3 * u_gmag0)
i_gq0 = -2 * q_g0 / (3 * u_gmag0)
u_cd0 = u_gmag0 - omega_g * L_f * i_gq0
u_cq0 = omega_g * L_f * i_gd0
omega_c0 = omega_g
u_c0 = I * Matrix([[u_cd0], [u_cq0]])
e_g0 = I * Matrix([[e_g0mag], [0]])
i_g0 = I * Matrix([[i_gd0], [i_gq0]])
u_g0 = e_g0

# Define controller parameters
Lhat = L_f
T_s = 1 / 10000
T_d = 1.5 * T_s
alpha_c = 2 * np.pi * 400
alpha_i = alpha_c
alpha_pll = 2 * np.pi * 20

# Calculate gains
k_p = (alpha_c + alpha_i) * Lhat
k_i = alpha_c * alpha_i * Lhat
k_t = alpha_c * Lhat

# Calculate auxiliary variables
G_theta = (
    1
    / u_gmag0
    * (2 * alpha_pll * s + alpha_pll**2)
    / (s**2 + 2 * alpha_pll * s + alpha_pll**2)
)
G_u = Matrix(
    [
        [
            2 * alpha_pll / (s + 2 * alpha_pll)
            - 1
            / u_gmag0
            * 2
            * alpha_pll
            / (s * (s + 2 * alpha_pll))
            * (k_i * i_gd0 - k_t * omega_c0 * i_gq0),
            -G_theta * (k_t / s * omega_c0 * i_gd0 + (k_p + k_i / s) * i_gq0 + u_cq0),
        ],
        [
            -1
            / u_gmag0
            * 2
            * alpha_pll
            / (s * (s + 2 * alpha_pll))
            * (k_t * omega_c0 * i_gd0 + k_i * i_gq0),
            G_theta * ((k_p + k_i / s) * i_gd0 - k_t / s * omega_c0 * i_gq0 + u_cd0),
        ],
    ]
)
G_i = (k_p + k_i / s) * I + k_t / s * omega_c0 * J
Z_f = s * L_f * I + omega_g * L_f * J

# Calculate output admittance
Y_c = (Z_f + exp(-s * T_d) * G_i) ** -1 * (I - exp(-s * T_d) * G_u)
Y_c = Y_c.subs(s, 1j * omega)

# Convert to numpy array
f_e = np.geomspace(1, 10e3, 100)
y_dd = lambdify(omega, Y_c[0, 0], "numpy")
y_qd = lambdify(omega, Y_c[0, 1], "numpy")
y_dq = lambdify(omega, Y_c[1, 0], "numpy")
y_qq = lambdify(omega, Y_c[1, 1], "numpy")

# Store result
res = IdentificationResults(
    f_e=f_e,
    Y_dd=y_dd(2 * np.pi * f_e),
    Y_qd=y_qd(2 * np.pi * f_e),
    Y_dq=y_dq(2 * np.pi * f_e),
    Y_qq=y_qq(2 * np.pi * f_e),
    i_g0=float(i_g0[0, 0]) + 1j * float(i_g0[1, 0]),
    e_g0=float(e_g0[0, 0]) + 1j * float(e_g0[1, 0]),
    u_g0=float(u_g0[0, 0]) + 1j * float(u_g0[1, 0]),
    u_c0=float(u_c0[0, 0]) + 1j * float(u_c0[1, 0]),
)

# Uncomment the row below to save results in a .csv file
# save_csv(res, "gfl_analytical_p0.5_q0.5")

plot_identification(res, "re_im", plot_passivity_index=True)
plot_vector_diagram(res, base)
