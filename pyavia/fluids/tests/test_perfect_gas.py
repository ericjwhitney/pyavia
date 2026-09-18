
import pytest

from .common_gas import (GasTestConfig, make_test_IDs, check_props,
                         check_correct_flow)
from .. import PerfectAir

# Tests for perfect gases.

# == Test Configurations ===============================================

# These are imported into specific suites as applicable.  Properties of
# particular note for each test are highlighted with '<-'.

# noinspection PyDictCreation
test_configs = {}

# TODO Check SI units correct and try others.
# TODO Test Supersonic Air
# TODO Test FAR


# -- 1 - Air, ISA S/L --------------------------------------------------

# ISA S/L air baseline reference conditions.
test_configs['Perfect - 1'] = GasTestConfig(
    gas='dry_air',
    description="Air, ISA S/L",
    units='SI',
    init_props={'p': 101_325, 'T': 288.15},  # [Pa] [K]
    check_props={
        'a'  : 340.293988 , # <- [m/s]
        'c_p': 1_004.68505, # <- [J/kg/K]
        'c_v': 717.632175 , # <- [J/kg/K]
        'M'  : 0.0        , # Check default.
        'p'  : 101_325    , # Check unchanged [Pa]
        'R'  : 287.05287  , # <- [J/kg/K]
        'T'  : 288.15     , # Check unchanged [K]
        'ρ'  : 1.22500002 , # <- [kg/m³]
        'γ'  : 1.4        , # <- Check default
        'μ'  : 1.7894e-5  , # <- [Pa.s]
    },
    flowing=False
)

# -- 2 - Air, Lab Conditions -------------------------------------------

# Check lab air reference state to verify 'h_ref', 's_ref'.
test_configs['Perfect - 2'] = GasTestConfig(
    gas='dry_air',
    description="Air, Lab Conds",
    units='SI',
    init_props={'p': 100_000, 'T': 298.15},  # [Pa] [K]
    check_props={
        'a'  : 346.148434 , # [m/s]
        'c_p': 1_004.68505, # [J/kg/K] Check no change.
        'c_v': 717.632175 , # [J/kg/K] Check no change.
        'h'  : 720_760    , # <- [J/kg] Reference datum.
        'M'  : 0.0        , # Check default.
        'R'  : 287.05287  , # [J/kg/K]  Check no change.
        's'  : 5_682.26   , # <- [J/kg/K]  Reference datum.
        'ρ'  : 1.16843160 , # [kg/m³]
        'γ'  : 1.4        , # Check default
    },
    flowing=False
)

# Check if (2) can be regenerated using 'p', 'h' (find 'T').
test_configs['Perfect - 2.1'] = GasTestConfig(
    gas='dry_air',
    description="Air, Lab Conds",
    units='SI',
    init_props={'h': 720_760, 'p': 100_000},  # [J/kg] [Pa]
    check_props={
        'a'  : 346.148434 , # [m/s]
        'c_p': 1_004.68505, # [J/kg/K] Check no change.
        'c_v': 717.632175 , # [J/kg/K] Check no change.
        'R'  : 287.05287  , # [J/kg/K] Check no change.
        's'  : 5_682.26   , # <- [J/kg/K] Reference datum.
        'T'  : 298.15     , # <- [Pa] [K]
        'ρ'  : 1.16843160 , # [kg/m³]
        'γ'  : 1.4        , # Check default
    },
    flowing=False
)

# Check if (2) can be regenerated using 'p', 's' (find 'T').
test_configs['Perfect - 2.2'] = GasTestConfig(
    gas='dry_air',  # Gas model.
    description="Air, Lab Conds",
    units='SI',
    init_props={'p': 100_000, 's': 5_682.26},  # [Pa] [J/kg/K]
    check_props={
        'a'  : 346.148434 , # [m/s]
        'c_p': 1_004.68505, # [J/kg/K] Check no change.
        'c_v': 717.632175 , # [J/kg/K] Check no change.
        'R'  : 287.05287  , # [J/kg/K]  Check no change.
        'h'  : 720_760    , # [J/kg] Reference datum.
        'T'  : 298.15     , # <- [Pa] [K]
        'ρ'  : 1.16843160 , # [kg/m³]
        'γ'  : 1.4        , # Check default
    },
    flowing=False
)

# Check if (2) can be regenerated using 'p', 'p0', 'h' (find 'M','T').
test_configs['Perfect - 2.3'] = GasTestConfig(
    gas='dry_air',  # Gas model.
    description="Air, Lab Conds",
    units='SI',
    init_props={
        'p' : 100_000, # [Pa]
        'p0': 100_000, # [Pa]
        'h' : 720_760, # [J/kg]
    },
    check_props={
        'a'  : 346.148434 , # [m/s]
        'c_p': 1_004.68505, # [J/kg/K] Check no change.
        'c_v': 717.632175 , # [J/kg/K] Check no change.
        'M'  : 0.0        , # <- Target.
        'R'  : 287.05287  , # [J/kg/K]  Check no change.
        's'  : 5_682.26   , # <- [J/kg/K] Reference datum.
        'T'  : 298.15     , # <- [Pa] [K]
        'ρ'  : 1.16843160 , # [kg/m³]
        'γ'  : 1.4        , # Check default
    },
    flowing=False
)

# --- 3. Hot Subsonic Air ----------------------------------------------

# Hot subsonic air (different γ).  Target values are for the perfect
# gas model; small changes are required for real gas models (see
# test_poly.py).
test_configs['Perfect - 3'] = GasTestConfig(
    gas='dry_air',
    description="Hot Subsonic Air",
    units='SI',
    init_props={
        'M': 0.75,
        'p': 1.0e5,      # [Pa]
        'T': 1_100,      # [K]
        'γ': 1.32921357  # <- Reduced
    },
    check_props={
        'a'  : 647.850313  , # [m/s]
        'c_p': 1_158.98797 , # [J/kg/K]
        'c_v': 871.935096  , # [J/kg/K]
        'h'  : 1_650_094.51, # [J/kg]
        'h0' : 1_768_137.96, # [J/kg]
        'M'  : 0.75        , # Check unchanged.
        'p'  : 100_000.000 , # [Pa]
        'p0' : 142_979.823 , # [Pa]
        'R'  : 287.05287   , # [J/kg/K]
        'T'  : 1_100.00    , # [K]
        'T0' : 1_201.85045 , # [K]
        'V'  : 485.887735  , # [m/s]
        'ρ'  : 0.316698073 , # [kg/m³]
        'γ'  : 1.32921357  , # Check unchanged.
    },
    flowing=True
)

# == Test Functions ====================================================

test_IDs = make_test_IDs(test_configs)


@pytest.mark.parametrize('test_config', test_configs.values(), ids=test_IDs)
def test_perfect_gas(test_config: GasTestConfig):
    match test_config.gas:
        case 'dry_air':
            gas = PerfectAir(**test_config.init_props)

        case _:
            raise ValueError(f"Unknown gas: {test_config.gas}")

    # Check target values are met.
    check_props(gas, test_config.check_props)

    # Check results are correct sense.
    if test_config.flowing is not None:
        check_correct_flow(gas, test_config.flowing)
