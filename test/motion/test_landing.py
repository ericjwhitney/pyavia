from unittest import TestCase

import numpy as np

# ======================================================================

typ_tyre_ft_lbf = np.array([
    # Tyre:      Goodyear 5.00-5, Type III, 4 Ply Nylon
    # Loading:   Dynamic
    # Inflation: 30 psig
    [0.000, 0],  # [in] (initially), [lbf]
    [0.535, 270],
    [0.910, 530],
    [1.180, 740],
    [1.300, 840],
    [1.460, 980],
    [1.650, 1160],
    [1.840, 1345],
    [2.020, 1525],
    [2.210, 1725],
    [2.455, 2000],
    [2.670, 2250],
    [2.825, 2500],
    [2.895, 2700]
])
typ_tyre_ft_lbf[:, 0] /= 12  # in -> [ft]


# ----------------------------------------------------------------------

class TestLandingEnergy(TestCase):
    STD_KWARGS = {
        'strut': 2158.3,  # Strut spring K [lbf/ft]
        'tyre': typ_tyre_ft_lbf,  # Tabulated (ft, lbf)
        'g': 32.174,  # [ft/s^2]
        'K': 2 / 3,  # L/W for light aircraft
        'm_ac': 1200 * 0.031081 / 2,  # lbm -> [slug]
        'V_sink': 7.0  # [fps]
    }

    def test_leaf_strut_tyre(self):
        from pyavia.motion.landing import landing_energy

        # Check results from standard case, with and without 5 lbm
        # tyre mass.
        res_no_tm = landing_energy(**self.STD_KWARGS, m_t=0)
        res_tm = landing_energy(**self.STD_KWARGS, m_t=5.0 * 0.031081)

        def all_almost_equal(name, value, places):
            self.assertAlmostEqual(getattr(res_no_tm, name), value,
                                   places=places)
            self.assertAlmostEqual(getattr(res_tm, name), value,
                                   places=places)

        # -- Check Constants -------------------------------------------

        all_almost_equal('W_ac', 600.0, 3)
        all_almost_equal('L', 400.0, places=3)
        all_almost_equal('V_sink', 7.0, places=6)
        all_almost_equal('g', 32.174, places=6)
        all_almost_equal('K', 2 / 3, places=6)

        # -- Approx. Equal Values --------------------------------------

        # Tyre stroke.
        self.assertAlmostEqual(res_no_tm.δ_t, 0.166, places=3)
        self.assertAlmostEqual(res_tm.δ_t, 0.166, places=3)

        # Tyre energy.
        self.assertAlmostEqual(res_no_tm.E_t, 107.6, places=1)
        self.assertAlmostEqual(res_tm.E_t, 107.5, places=1)

        # Tyre force.
        self.assertAlmostEqual(res_no_tm.F_t, 1500.5, places=1)
        self.assertAlmostEqual(res_tm.F_t, 1499.9, places=1)

        # -- Distinct Values -------------------------------------------

        # Strut stroke.
        self.assertAlmostEqual(res_no_tm.δ_s, 0.695, places=3)
        self.assertAlmostEqual(res_tm.δ_s, 0.693, places=3)

        # Strut energy.
        self.assertAlmostEqual(res_no_tm.E_s, 521.6, places=1)
        self.assertAlmostEqual(res_tm.E_s, 517.7, places=1)

        # Strut force.
        self.assertAlmostEqual(res_no_tm.F_s, 1500.5, places=1)
        self.assertAlmostEqual(res_tm.F_s, 1494.9, places=1)
