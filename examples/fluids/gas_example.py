#!/usr/bin/env python3

# Comparison of gas models.  SI Units.
# Written by: Eric J. Whitney  Last updated: September 2026.

from pyavia.fluids import PerfectAir, PolyAir

T      = 200      # [K]
p      = 101_325  # [Pa]
M      = 0.5
T_step = 50       # [K]

# PerfectAir default γ = 1.4.

prop_list = ['T0', 'p0', 'h', 's']
units = ['K', 'Pa', 'J/kg', 'J/kg/K']

print(f"\nComparison vs. T for gas models with  P = {p:.5G}, M = {M:.5G}\n")
while True:  # Until model fails.
    try:
        real = PolyAir(T=T, p=p, M=M)
        perfect = PerfectAir(T=T, p=p, M=M)
        print(f"T={T:6.5G}", end='')
        for prop, unit in zip(prop_list, units):
            real_x = getattr(real, prop)
            perf_x =  getattr(perfect, prop)
            err: float = (perf_x - real_x) / real_x * 100.0
            print(f" | {prop:s}={real_x:#.5G} (Real) vs. "
                  f"{perf_x:#.5G} (Perf) Err={err:+.1f}%", end='')
        print()
        T += T_step

    except (RuntimeError, ValueError) as ex:
        print(f"\nStopped -> {ex}")
        break
