from unittest import TestCase


class TestLinearInterp(TestCase):
    def test_linear_interp(self):
        from pyavia.numeric.interpolate import linear_int_ext

        # Also checks line_pt by proxy.
        data = [(9, 100000),
                (8, 10000),
                (7, 1000)]

        # Linear interp.
        res = linear_int_ext(data, (7.5, None))
        self.assertAlmostEqual(res[1], 5500)

        # Scaled linear.
        res = linear_int_ext(data, (None, 31622.7766016),
                             scale=(None, 'log'))
        self.assertAlmostEqual(res[0], 8.5)

        # Out of bounds if no extrap.
        with self.assertRaises(ValueError):
            linear_int_ext(data, (6, None))

        # High side extrap.
        res = linear_int_ext(data, (None, 190000),
                             allow_extrap=True)
        self.assertAlmostEqual(res[0], 10)

        # Low side extrap, log scale.
        res = linear_int_ext(data, (6, None), scale=(None, 'log'),
                             allow_extrap=True)
        self.assertAlmostEqual(res[1], 100)
