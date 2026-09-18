#!/usr/bin/env python3

# Compute a simple nozzle from given chamber (resevoir) and exit conditions,
# assuming a constantly increasing Mach number.  If the chamber pressure to
# exit pressure ratio is sufficiently high the nozzle will automatically choke
# and a converging-diverging shape will be produced.

# Written by: Eric J. Whitney  Last updated: 5 January 2020.

import time
import matplotlib.pyplot as plt

from pyavia.fluids import PolyAirFlow, make_gas
from pyavia.units import Dim

# Create flowing version of (imperfect) gas model PolyAir.


P_ATM = 101_325  # [Pa]

chamber = PolyAirFlow(
    T     = 700,        # [K]
    p     = 25 * P_ATM, # [Pa]
    M     = 0.1,
    m_dot = 10,         # [kg/s]
    # FAR   = 0.01,
    # fuel  = 'kerosene'  # FutureWork: Passing this argument is a problem.
)
p_exit = P_ATM

M, ΔM = chamber.M, 0.01
M_x, p_p0_x, T_T0_x, A_x = [], [], [], []
last_gas = None

print(f"Computing Nozzle.\n")
print(f"{'M':>14s}{'P/P0':>14s}{'T/T0':>14s}{'A':>14s}")
t_start = time.time()
while True:
    # Basic Method: This method may involve some more calculation cost to
    # converge to the correct properties
    # local_gas = PolyAirFlow(m_dot=chamber.m_dot, FAR=chamber.FAR,
    #                         h0=chamber.h0, s=chamber.s, M=M)

    # Advanced Method: Use the gas from the previous iteration to
    # quickly find the properties here.

    # FutureWork: Probably better to make this a more generic
    # 'fit_model_lsq' type approach which would better handle
    # different arguments.
    local_gas = make_gas(
        PolyAirFlow,
        init_props=('p', 'T', 'M', 'm_dot', 'FAR'),
        ref_gas=last_gas,
        h0=chamber.h0, s=chamber.s, M=M, m_dot=chamber.m_dot,
        # fuel=chamber.fuel,  # FutureWork: Problem argument.
        FAR=chamber.FAR
    )

    M_x.append(M)
    p_p0_x.append(local_gas.p / chamber.p0)
    T_T0_x.append(local_gas.T / chamber.T0)
    A_x.append(local_gas.m_dot / local_gas.ρ / local_gas.V)
    print(f"{M_x[-1]:14.4f}{p_p0_x[-1]:14.4f}"
          f"{T_T0_x[-1]:14.4f}{A_x[-1]:14.4f}")
    if local_gas.p < p_exit:
        break
    M += ΔM
    last_gas = local_gas

At = min(A_x)
A_At_x = [A / At for A in A_x]

Me, Ae_At = M_x[-1], A_At_x[-1]
pe_p0, Te_T0 = p_p0_x[-1], T_T0_x[-1]
γ = chamber.γ
pred_pe_p0 = (1 + 0.5 * (γ - 1) * Me ** 2) ** (-γ / (γ - 1))
pred_Te_T0 = 1 / (1 + 0.5 * (γ - 1) * Me ** 2)
t_end = time.time()

print(f"\nAe/A* = {A_At_x[-1]:.4f}.")
print(f"Using chamber gamma = {γ:.3f} and exit Mach number Me = "
      f"{M_x[-1]:.4f}:")
print(f"\tPredicted Pe/P0 = {pred_pe_p0:.4f} vs. Computed Pe/P0 = {pe_p0:.4f}")
print(f"\tPredicted Te/T0 = {pred_Te_T0:.4f} vs. Computed Te/T0 = {Te_T0:.4f}")
print(f"\tSolution took {t_end - t_start:.3f} seconds.")

plt.figure()
plt.xlabel("$M$")
plt.ylabel("$P/P_0$, $T/T_0$")
plt.ylim((0, 1))
plt.grid()
plt.plot(M_x, p_p0_x, 'b', label="$P/P_0$")
plt.plot(M_x, T_T0_x, 'r', label="$T/T_0$")
plt.legend(loc='upper right')

plt.figure()
plt.xlabel("$M$")
plt.ylabel("$A/A*$")
plt.grid()
plt.plot(M_x, A_At_x, color='k')
plt.show()
