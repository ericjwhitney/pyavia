from unittest import TestCase


class TestLinearInterp(TestCase):
    def test_linear_interp(self):
        from pyavia.data import linear_int_ext

        # Also checks line_pt by proxy.

        data = [(9, 100000),
                (8, 10000),
                (7, 1000)]

        res = linear_int_ext(data, (7.5, None))  # Linear interp.
        self.assertAlmostEqual(res[1], 5500)

        res = linear_int_ext(data, (None, 31622.7766016),  # Scaled linear.
                             scale=(None, 'log'))
        self.assertAlmostEqual(res[0], 8.5)

        with self.assertRaises(ValueError):
            linear_int_ext(data, (6, None))  # Out of bounds if no extrap.

        res = linear_int_ext(data, (None, 190000),
                             allow_extrap=True)  # High side extrap.
        self.assertAlmostEqual(res[0], 10)

        res = linear_int_ext(data, (6, None), scale=(None, 'log'),
                             allow_extrap=True)  # Low side extrap, log scale.
        self.assertAlmostEqual(res[1], 100)
