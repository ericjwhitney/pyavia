import dataclasses

import numpy as np
import pytest

from .common_gas import (make_test_IDs, GasTestConfig, check_props,
                         check_correct_flow)
from .test_perfect_gas import test_configs as perfect_configs
from .. import PolyAir

# Tests for polynomial-model (real) gases.


# == Test Configurations ===============================================

# These are imported into specific suites # as applicable.  Properties
# of particular note for each test are # highlighted with '<-'.

# TODO Check SI units correct and try others.
# TODO Test FAR
# TODO Test Supersonic Air

# -- 1, 2 - Air, ISA S/L and Lab Conds ---------------------------------

# Tests 1-2 cloned from from perfect gas tests.
# noinspection PyDictCreation
test_configs = {
    'Poly - 1'  : perfect_configs['Perfect - 1'],
    'Poly - 2'  : perfect_configs['Perfect - 2'],
    'Poly - 2.1': perfect_configs['Perfect - 2.1'],
    'Poly - 2.2': perfect_configs['Perfect - 2.2'],
    'Poly - 2.3': perfect_configs['Perfect - 2.3'],
}

# --- 3. Hot Subsonic Air ----------------------------------------------

# Test 3 repeated from perfect gas tests with minor changes needed due
# to differences between perfect gas and polynomial gas models.
test_configs['Poly - 3'] = dataclasses.replace(
        perfect_configs['Perfect - 3'],
        init_props={k: perfect_configs['Perfect - 3'].init_props[k]
                    for k in ('p', 'T', 'M')},  # 'γ' not reqd for init.
        check_props=perfect_configs['Perfect - 3'].check_props | {
            'h': 1_583_453.59,   # [J/kg] Replacement values are ...
            'h0': 1_701_497.04,  # [J/kg] ... slightly different ...
            'p0': 142_989.941,   # [Pa] ... for polynomial model.
            'T0': 1_201.14747,   # [K]
        },
)

# --- 4. Thermodynamic Chart, Air --------------------------------------


_IDEAL_AIR_REF = np.array([
    # Ideal-gas properties of air from Cengel Y. and Boles, M.,
    # "Thermodynamics: An Engineering Approach", 1998.
    # Table A-2 ->              Table A-17 ->
    #  T    c_p    c_v     γ        h     Pr       u      vr       s°
    # [K]  [kJ/   [kJ/    ---     [kJ/    ---    [kJ/     ---    [kJ/
    #       kg/K]  kg/K]            kg]            kg]            kg/K]
    [ 250, 1.003, 0.716, 1.401,  250.05, 0.7329, 178.28, 979.0, 1.51917],
    [ 300, 1.005, 0.718, 1.400,  300.19, 1.3860, 214.07, 621.2, 1.70203],
    [ 350, 1.008, 0.721, 1.398,  350.49, 2.379,  250.02, 422.2, 1.85708],
    [ 400, 1.013, 0.726, 1.395,  400.98, 3.806,  286.16, 301.6, 1.99194],
    [ 450, 1.020, 0.733, 1.391,  451.80, 5.775,  322.62, 223.6, 2.11161],
    [ 500, 1.029, 0.742, 1.387,  503.02, 8.411,  359.49, 170.6, 2.21952],
    [ 550, 1.040, 0.753, 1.381,  555.74, 11.86,  396.86, 133.1, 2.31809],
    [ 600, 1.051, 0.764, 1.376,  607.02, 16.28,  434.78, 105.8, 2.40902],
    [ 650, 1.063, 0.776, 1.370,  659.84, 21.86,  473.25, 85.34, 2.49364],
    [ 700, 1.075, 0.788, 1.364,  713.27, 28.80,  512.33, 69.76, 2.57277],
    [ 750, 1.087, 0.800, 1.359,  767.29, 37.35,  551.99, 57.63, 2.64737],
    [ 800, 1.099, 0.812, 1.354,  821.95, 47.75,  592.30, 48.08, 2.71787],
    [ 900, 1.121, 0.834, 1.344,  932.93, 75.29,  674.58, 34.31, 2.84856],
    [1000, 1.142, 0.855, 1.336, 1046.04, 114.0,  758.94, 25.17, 2.96770],
])

test_configs['Poly - 4'] = GasTestConfig(
    gas='dry_air',
    description="Thermodynamic Charts Air",
    units='SI',
    init_props={
        'p': 101_325,               # [Pa] (1 atm ref. pressure).
        'T': _IDEAL_AIR_REF[:, 0],  # [K]
    },
    check_props={
        'c_p': _IDEAL_AIR_REF[: , 1] * 1000, # kJ/kg/K -> [J/kg/K]
        'c_v': _IDEAL_AIR_REF[: , 2] * 1000, # kJ/kg/K -> [J/kg/K]
        'γ'  : _IDEAL_AIR_REF[: , 3],
    },
    flowing=False
)

# TODO Try to solve given 'h', 's', 'M' for the table.  This checks
#  least squares convergence over a large number of points.


# == Test Functions ====================================================

test_IDs = make_test_IDs(test_configs)


@pytest.mark.parametrize('test_config', test_configs.values(), ids=test_IDs)
def test_poly_gas(test_config: GasTestConfig):
    match test_config.gas:
        case 'dry_air':
            gas = PolyAir(**test_config.init_props)

        case _:
            raise ValueError(f"Unknown gas: {test_config.gas}")

    # Check target values are met.
    check_props(gas, test_config.check_props)

    # Check results are correct sense.
    if test_config.flowing is not None:
        check_correct_flow(gas, test_config.flowing)
