#!/usr/bin/env python3

# Generate examples of Q-tables using a selected GasFlow object.
# Written by: Eric J. Whitney  Last updated: 8 January 2020.

from math import isclose
from pyavia.aero import PerfectGasFlow, GasFlowWF
from pyavia import Dim

M, M_stop, M_step = 0.00, 2.50, 0.05
use_gas = 'real_ssl_air'

P_ref = Dim(1.0, 'atm')  # SSL reference conditions with unit massflow.
T_ref = Dim(288.15, 'K')
w_ref = Dim(1.0, 'kg/s')

if use_gas == 'perfect_cold_air':
    gas_model = PerfectGasFlow(gamma=1.4, gas='air',
                               P=P_ref, T=T_ref, M=M, w=w_ref)
elif use_gas == 'perfect_hot_air':
    gas_model = PerfectGasFlow(gamma=1.33, gas='air',
                               P=P_ref, T=T_ref, M=M, w=w_ref)
elif use_gas == 'real_ssl_air':
    gas_model = GasFlowWF(gas='air', FAR=0.0,
                          P=P_ref, T=T_ref, M=M, w=w_ref)
else:
    raise ValueError(f"Unknown gas: {use_gas}")

vt_units = '        m.s⁻¹.K⁻⁰ᐧ⁵'
q_units = '  kg.K⁰ᐧ⁵/m²/kPa/s'
Q_units = '   kg.K⁰ᐧ⁵/m²/kPa/s'

print(f"\nQ-Curve Data - Reference Flow -> {gas_model}")

col_titles = ['Mach No.', 'P0/P', '(P0-P)/P0', 'T0/T', 'V/√T', 'q', 'Q']
print(''.join(f'{x:>18s}' for x in col_titles))
col_units = ['---', '---', '%', '---', vt_units, q_units, Q_units]
print(''.join(f'{x:>18}' for x in col_units))

while M < M_stop or isclose(M, M_stop):
    gas = gas_model.new_props(T=gas_model.T, P=gas_model.P, M=M)

    vt = (gas.u / (gas.T0 ** 0.5)).convert(vt_units)
    # Rearrange Q = 1000 * W * (T0 ** 0.5) / (A * P0):
    #   -> Q = rho * V * (T0 ** 0.5) / P0
    Q = (gas.rho * gas.u * gas.T0 ** 0.5 / gas.P0).convert(Q_units)
    q = ((gas.P0/gas.P) * Q).convert(q_units)
    print(f"{M:18.2f}{gas.P0/gas.P:18.4f}{(gas.P0-gas.P)/gas.P0*100.0:18.4f}"
          f"{gas.T0/gas.T:18.4f}{float(vt):18.4f}{float(q):18.4f}"
          f"{float(Q):18.4f}")

    M += M_step
