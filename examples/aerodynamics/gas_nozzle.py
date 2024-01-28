#!/usr/bin/env python3

# Compute a simple nozzle from given chamber (resevoir) and exit conditions,
# assuming a constantly increasing Mach number.  If the chamber pressure to
# exit pressure ratio is sufficiently high the nozzle will automatically choke
# and a converging-diverging shape will be produced.

# Written by: Eric J. Whitney  Last updated: 5 January 2020.

import time
import matplotlib.pyplot as plt
from pyavia.aerodynamics.imperfect_gas import ImperfectGas
from pyavia.units import Dim

chamber = ImperfectGas(T=Dim(700, 'K'), P=Dim(25.0, 'atm'), M=0.1,
                       w=Dim(10, 'kg/s'), gas='air', FAR=0)
P_exit = Dim(1.0, 'atm')

M, M_step = chamber.M, 0.01
Mx, PP0x, TT0x, Ax = [], [], [], []
last_gas = None

print(f"Computing Nozzle.\n")
print(f"{'M':>14s}{'P/P0':>14s}{'T/T0':>14s}{'A':>14s}")
t_start = time.time()
while True:
    local_gas = ImperfectGas(w=chamber.w, gas=chamber.gas, FAR=chamber.FAR,
                             h0=chamber.h0, s=chamber.s, M=M,
                             init_gas=last_gas)
    Mx.append(M)
    PP0x.append(local_gas.P / chamber.P0)
    TT0x.append(local_gas.T / chamber.T0)
    Ax.append(local_gas.w / local_gas.rho / local_gas.u)
    print(f"{Mx[-1]:14.4f}{PP0x[-1]:14.4f}{TT0x[-1]:14.4f}{Ax[-1]:14.4f}")
    if local_gas.P < P_exit:
        break
    M += M_step
    last_gas = local_gas

At = min(Ax)
AAtx = [A / At for A in Ax]

Me, AeAt = Mx[-1], AAtx[-1]
PeP0, TeT0 = PP0x[-1], TT0x[-1]
gamma = chamber.gamma
predPeP0 = (1 + 0.5 * (gamma - 1) * Me ** 2) ** (-gamma / (gamma - 1))
predTeT0 = 1 / (1 + 0.5 * (gamma - 1) * Me ** 2)
t_end = time.time()

print(f"\nAe/A* = {AAtx[-1]:.4f}.")
print(f"Using chamber gamma = {gamma:.3f} and exit Mach number Me = "
      f"{Mx[-1]:.4f}:")
print(f"\tPredicted Pe/P0 = {predPeP0:.4f} vs. Computed Pe/P0 = {PeP0:.4f}")
print(f"\tPredicted Te/T0 = {predTeT0:.4f} vs. Computed Te/T0 = {TeT0:.4f}")
print(f"\tSolution took {t_end - t_start:.3f} seconds.")

plt.figure()
plt.xlabel("$M$")
plt.ylabel("$P/P_0$, $T/T_0$")
plt.ylim([0, 1])
plt.grid()
plt.plot(Mx, PP0x, 'b', label="$P/P_0$")
plt.plot(Mx, TT0x, 'r', label="$T/T_0$")
plt.legend(loc='upper right')

plt.figure()
plt.xlabel("$M$")
plt.ylabel("$A/A*$")
plt.grid()
plt.plot(Mx, AAtx, color='k')
plt.show()
