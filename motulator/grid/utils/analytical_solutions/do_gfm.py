"""Script for computing analytical output admittance using DO-GFM control."""

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
nom = utils.NominalValues(U=400, I=18, f=50, P=12.5e3)
base = utils.BaseValues.from_nominal(nom)

# Model parameters
e_g0mag = base.u
omega_g = base.w
L_g = 0.74 * base.L
L_f = 0.15 * base.L

# Define the operating-point grid active power and converter output voltage magnitude
p_g0 = 0.5 * base.p
u_c0mag = base.u

# Calculate other operating-point quantities
delta = np.arcsin(2 * omega_g * (L_f + L_g) * p_g0 / (3 * u_c0mag * e_g0mag))
phi_g = np.arctan(
    L_g * u_c0mag * np.sin(delta) / (L_g * u_c0mag * np.cos(delta) + L_f * e_g0mag)
)
u_c0 = I * Matrix(
    [[u_c0mag * np.cos(delta - phi_g)], [u_c0mag * np.sin(delta - phi_g)]]
)
e_g0 = Matrix(
    [[np.cos(phi_g), np.sin(phi_g)], [-np.sin(phi_g), np.cos(phi_g)]]
) * Matrix([[e_g0mag], [0]])
i_g0 = (omega_g * (L_f + L_g) * J) ** -1 * (u_c0 - e_g0)
u_g0 = e_g0 + omega_g * L_g * J * i_g0
vhat_c0 = u_c0
vhat_c0mag = u_c0mag

# Define controller parameters
R_a = 0.2 * base.Z
k_vmag = 1
alpha_o = 1 * base.w
Lhat_t = L_f
omegahat_g = omega_g
T_s = 1 / 10000
T_d = 1.5 * T_s

# Calculate gains
k_p0 = 2 * R_a / (3 * vhat_c0mag**2) * vhat_c0
k_v0 = (I - k_vmag * J) / vhat_c0mag * vhat_c0
K_o = alpha_o * I - omegahat_g * J

# Calculate auxiliary variables
K_1 = 3 / 2 * k_p0 * i_g0.T + 1 / vhat_c0mag * k_v0 * vhat_c0.T
K_2 = 3 / 2 * k_p0 * vhat_c0.T
G_i = (I - K_1) * (s * I + (omegahat_g * J + K_o) * K_1) ** -1 * (
    (omegahat_g * J + K_o) * K_2 + s * Lhat_t * K_o
) + K_2
Z_f = s * L_f * I + omega_g * L_f * J

# Calculate inverse of output admittance matrix
Y_c_inv = Z_f + exp(-s * T_d) * G_i
Y_c_inv = Y_c_inv.subs(s, 1j * omega)

# Convert to numpy array and compute output admittance
Y_c_inv_lambda = lambdify(omega, Y_c_inv, "numpy")
f_e = np.geomspace(1, 10e3, 100)
y_c_inv = Y_c_inv_lambda(2 * np.pi * f_e)
y_c_inv = np.moveaxis(y_c_inv, -1, 0)
y_c = np.linalg.inv(y_c_inv)

# Store result
res = IdentificationResults(
    f_e=f_e,
    Y_dd=y_c[:, 0, 0],
    Y_qd=y_c[:, 0, 1],
    Y_dq=y_c[:, 1, 0],
    Y_qq=y_c[:, 1, 1],
    i_g0=float(i_g0[0, 0]) + 1j * float(i_g0[1, 0]),
    e_g0=float(e_g0[0, 0]) + 1j * float(e_g0[1, 0]),
    u_g0=float(u_g0[0, 0]) + 1j * float(u_g0[1, 0]),
    u_c0=float(u_c0[0, 0]) + 1j * float(u_c0[1, 0]),
)

# Uncomment the row below to save results in a .csv file
# save_csv(res, "do-gfm_analytical_p0.5_Lg0.74")

plot_identification(res, "re_im", plot_passivity_index=True)
plot_vector_diagram(res, base)
