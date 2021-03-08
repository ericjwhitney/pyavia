#!/usr/bin/env python3

# Comparison of gas models.
# Written by: Eric J. Whitney  Last updated: 28 December 2019.

from pyavia.aero import PerfectGasFlow, GasFlowWF, GasError
from pyavia import Dim

T, T_step = Dim(200, 'K'), Dim(100, 'K')
P = Dim(1.0, 'atm')
M = 0.5

prop_list = ['T0', 'P0', 'h', 's']
units = ['K', 'kPa', 'kJ/kg', 'kJ/kg/K']

print(f"\nComparison vs. T for Gas Models with  P = {P:.5G}, M = {M:.5G}\n")
while True:  # Until model fails.
    try:
        real = GasFlowWF(T=T, P=P, M=M, gas='air')
        perfect = PerfectGasFlow(T=T, P=P, M=M, gamma=1.4)
        print(f"T = {T:.5G}", end='')
        for prop, unit in zip(prop_list, units):
            real_x, perf_x = getattr(real, prop), getattr(perfect, prop)
            err = (perf_x - real_x) / real_x * 100.0
            print(f"\t {prop}: Real {real_x.convert(unit):.5G}"
                  f", Perfect {perf_x.convert(unit):.5G}"
                  f" (Error {err:+.2f} %)", end='')
        print()
        T += T_step

    except GasError as ex:
        print(f"\nStopped -- {ex}")
        break
