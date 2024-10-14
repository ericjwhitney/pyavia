from unittest import TestCase


# ======================================================================

class TestForceType(TestCase):
    def test_force_type(self):
        from pyavia.util.type_ops import force_type

        x = force_type(3.5, int, float)
        self.assertIsInstance(x, int)  # -> 3 (int)
        self.assertEqual(x, 3)

        x = force_type("3.5+4j", float, complex)
        self.assertIsInstance(x, complex)  # -> 3.5+4j (complex)
        self.assertEqual(x, 3.5 + 4j)

        x = force_type(3.5 + 4j, int, float, str)
        self.assertIsInstance(x, str)
        self.assertEqual(x, '(3.5+4j)')  # -> '(3.5+4j)' (str)

        with self.assertRaises(ValueError):
            force_type(3.5 + 4j, int, float)


# ----------------------------------------------------------------------

class TestCoaxType(TestCase):
    def test_coax_type(self):
        from pyavia.util.type_ops import coax_type

        x = coax_type(3.5, int, float)
        self.assertIsInstance(x, float)  # Because int(3.5) != 3.5
        self.assertEqual(x, 3.5)

        x = coax_type(3.0, int, str)
        self.assertIsInstance(x, int)
        self.assertEqual(x, 3)

        with self.assertRaises(ValueError):
            coax_type("3.0", int, float)  # Because '3.0' != 3.0

        x = 3 + 2j
        y = coax_type(x, int, float, default=x)
        self.assertIsInstance(y, complex)
        self.assertEqual(y, 3 + 2j)

# ======================================================================
