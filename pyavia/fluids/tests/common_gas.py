"""Common functions for testing gases."""
from dataclasses import dataclass
from functools import partial

import pytest
from numpy import typing as npt

from .. import Gas

# Original by Eric J. Whitney, December 2020.

# ======================================================================

# Define common dataclass for testing different gases.
@dataclass(frozen=True)
class GasTestConfig:
    gas: str
    description: str
    units: str
    init_props: dict[str, npt.ArrayLike]
    check_props: dict[str, npt.ArrayLike]
    flowing: bool | None


# == Test Utilities ====================================================

# Setup a 'numerically equal' tolerance, i.e. within roundoff /
# algorithm error.
eq_numeric = partial(pytest.approx, abs=1e-8, rel=1e-8)

# Setup 'engineering' tolerances for checking gas properties against
# reference values. These are needed as different gas models or
# convergence algorithms can give slightly difference results due to
# fitting errors, etc.  Relative tolerances are generally found by
# applying the absolute tolerance to reference values for air in SI
# units.

# Specific enthalpy: ±1.0 J/kg or lab air 'h' relative equiv.
eq_h = partial(pytest.approx, abs=1.0, rel=1.5e-6)

# Specific entropy: ±0.1 J/kg/K or lab air 's' relative equiv.
eq_s = partial(pytest.approx, abs=0.1, rel=2.0e-5)

# Specific heat: ±1.5 J/kg/K or ISA air 'c_p' relative equiv.
eq_c = partial(pytest.approx, abs=1.5, rel=1.5e-3)

# Mach No.: ±0.001.
eq_M = partial(pytest.approx, abs=0.001)

# Pressure: ±1 Pa or ISA air relative equiv.
eq_p = partial(pytest.approx, abs=1.0, rel=1.0e-5)

# Temperature: ±0.01 K or ISA air relative equiv.
eq_T = partial(pytest.approx, abs=0.01, rel=3.5e-5)

# Speed: ±0.1 m/s or ISA air 'a' relative equiv.
eq_V = partial(pytest.approx, abs=0.1, rel=3.0e-4)

# Ratio of spec. heats: ±0.001.
eq_γ = partial(pytest.approx, abs=0.001)

# Density: ±0.0001 kg/m³ or ISA air 'ρ' relative equiv.
eq_ρ = partial(pytest.approx, abs=0.0001, rel=8.5e-5)

# Dynamic viscosity: ±0.01e-5 Pa.s or ISA air 'μ' relative equiv.
eq_μ = partial(pytest.approx, abs=1.0e-7, rel=5.0e-3)

# Other properties (UNO): ±0.0001 or 1 part per million.
eq_other = partial(pytest.approx, abs=0.0001, rel=1.0e-6)


# ----------------------------------------------------------------------


def check_props(gas: Gas, prop_targets: dict[str, npt.ArrayLike]):
    # This function checks request properties of a Gas object, applying
    # appropriate checking tolerance depending on the property.
    for prop, target in prop_targets.items():
        value = getattr(gas, prop)

        match prop:
            case 'a' | 'V':
                assert value == eq_V(target)

            case 'c_p' | 'c_v':
                assert value == eq_c(target)

            case 'h' | 'h0':
                assert value == eq_h(target)

            case 'M':
                assert value == eq_M(target)

            case 's' | 's0':
                assert value == eq_s(target)

            case 'T' | 'T0':
                assert value == eq_T(target)

            case 'γ':
                assert value == eq_γ(target)

            case 'ρ':
                assert value == eq_ρ(target)

            case 'μ':
                assert value == eq_μ(target)

            case _:  # Default.
                assert value == eq_other(target)


# ----------------------------------------------------------------------


def check_correct_flow(gas: Gas, is_flowing: bool):
    # This function checks overall correct behaviour of a flow
    # depending on whether it is flowing or stationary.

    if is_flowing:
        assert gas.V > 0.0
        assert gas.M > 0.0
        assert gas.h < gas.h0  # Stagnation values higher.
        assert gas.p < gas.p0
        assert gas.T < gas.T0
    else:
        assert gas.V == eq_V(0.0)
        assert gas.M == eq_M(0.0)
        assert gas.M >= 0.0  # Also check no tiny M < 0.
        assert gas.h == eq_h(gas.h0)  # Stagnation values equal.
        assert gas.p == eq_p(gas.p0)
        assert gas.T == eq_T(gas.T0)


# ----------------------------------------------------------------------

def make_test_IDs(configs: dict[str, GasTestConfig]) -> list[str]:
    return [f"{k} - {config.description} - "
            f"({', '.join(config.init_props.keys())}) - "
            f"{config.units}"
            for k, config in configs.items()]
