#!/usr/bin/env python3

# Generate examples of Q-tables using a selected GasFlow object.
# Written by: Eric J. Whitney  Last updated: 15 January 2022.

from math import isclose

from pyavia.fluids import PerfectAirFlow, PolyAirFlow

M, M_stop, ΔM = 0.00, 2.50, 0.05
use_gas = 'real_ssl_air'

# SSL reference conditions with unit massflow.
p_ref = 101_325  # [Pa]
T_ref = 288.15   # [K]
m_dot_ref = 1.0      # [kg/s]

gas_models = {
    'perfect_cold_air': {
        'model': PerfectAirFlow,
        'kwargs': {'γ': 1.4}},

    'perfect_hot_air': {
        'model': PerfectAirFlow,
        'kwargs': {'γ': 1.33}},

    'real_ssl_air': {
        'model': PolyAirFlow,
        'kwargs': {'FAR': 0.0}}
}

vt_units = 'm.s⁻¹/√K'
q_units = 'kg.√K/m²/Pa/s'
Q_units = 'kg.√K/m²/Pa/s'

print(f"\nQ-Curve Data - Reference Flow -> {use_gas}")

col_titles = ['Mach No.', 'P0/P', '(P0-P)/P0', 'T0/T', 'V/√T', 'q', 'Q']
print(''.join(f'{x:>18s}' for x in col_titles))
col_units = ['---', '---', '%', '---', vt_units, q_units, Q_units]
print(''.join(f'{x:>18s}' for x in col_units))

while M < M_stop or isclose(M, M_stop):
    gas_model = gas_models[use_gas]['model']
    kwargs = gas_models[use_gas]['kwargs']
    gas = gas_model(T=T_ref, p=p_ref, M=M, m_dot=m_dot_ref, **kwargs)

    vt = gas.V / (gas.T0 ** 0.5)

    # Rearrange Q = 1000 * W * (T0 ** 0.5) / (A * P0):
    #        -> Q = ρ * V * (T0 ** 0.5) / p0
    Q = gas.ρ * gas.V * gas.T0 ** 0.5 / gas.p0
    q = Q * (gas.p0 / gas.p)

    print(
        f"{M:18.2f}"
        f"{gas.p0 / gas.p:18.4f}"
        f"{float((gas.p0 - gas.p) / gas.p0 * 100.0):18.4f}"
        f"{gas.T0 / gas.T:18.4f}"
        f"{float(vt):18.4f}"
        f"{float(q):18.4f}"
        f"{float(Q):18.4f}"
    )

    M += ΔM
