from unittest import TestCase


class TestFortranArray(TestCase):
    def test__init__(self):
        from pyavia import FortranArray
        import numpy as np

        # Test basic initialisation.
        row_mat = FortranArray([1.0, 2.0, 3.0, 4.0])  # Row vector, float64.
        col_mat = FortranArray([[1], [2], [3], [4]])  # Column vector, int.
        full_mat = FortranArray([[3, 6], [-5, -9]], dtype=np.float64)

        self.assertEqual(row_mat.dtype, np.float64)
        self.assertEqual(row_mat.shape, (4,))

        self.assertEqual(col_mat.dtype, np.int32)
        self.assertEqual(col_mat.shape, (4, 1))

        self.assertEqual(full_mat.dtype, np.float64)
        self.assertEqual(full_mat.shape, (2, 2))

    def test__getitem__(self):
        from pyavia import FortranArray
        import numpy as np

        # Test Fortran individual indexing.
        row_mat = FortranArray([1.0, 2.0, 3.0, 4.0])  # Row vector, float64.
        col_mat = FortranArray([[1], [2], [3], [4]])  # Column vector, int.
        full_mat = FortranArray([[3, 6], [-5, -9]], dtype=np.float64)

        self.assertEqual(row_mat[2], 2.0)
        self.assertEqual(row_mat[4].dtype, np.float64)
        with self.assertRaises(TypeError):
            print(f"Wrong dims = {row_mat[1, 1]}")  # Array only 1-D.
        with self.assertRaises(IndexError):
            print(f"Negative blocked = {row_mat[-1]}")
        with self.assertRaises(IndexError):
            print(f"Nonexistant item = {row_mat[0]}")
        with self.assertRaises(IndexError):
            print(f"Nonexistant item = {row_mat[5]}")

        self.assertEqual(col_mat[4, 1], 4)
        self.assertEqual(col_mat[1, 1].dtype, np.int32)
        with self.assertRaises(TypeError):
            print(f"Wrong dims = {col_mat[1]}")  # Array is 2-D.
        with self.assertRaises(IndexError):
            print(f"Nonexistant item = {col_mat[0, 1]}")
        with self.assertRaises(IndexError):
            print(f"Nonexistant item = {col_mat[4, 2]}")

        self.assertEqual(full_mat[1, 2], 6.0)
        self.assertEqual(full_mat[2, 2].dtype, np.float64)
        with self.assertRaises(TypeError):
            print(f"Wrong dims = {full_mat[1, 1, 1]}")
        with self.assertRaises(IndexError):
            print(f"Nonexistant item = {full_mat[0, 0]}")
        with self.assertRaises(IndexError):
            print(f"Nonexistant item = {full_mat[2, 3]}")

        # Test Fortran slicing.
        self.assertTrue(all(row_mat[2:3] == [2.0, 3.0]))
        self.assertTrue(all(row_mat[1:8:3] == [1.0, 4.0]))  # Legal in F90.
        self.assertTrue(all(col_mat[4:3:-1, 1] == [4.0, 3.0]))
        self.assertTrue(np.all(col_mat[2::-1, :] == np.array([[2.0], [1.0]])))
        self.assertTrue(np.all(full_mat[::-1, ::-1] == np.flip(full_mat)))

    def test__setitem__(self):
        from pyavia import FortranArray
        import numpy as np

        # Test slice indexing.
        r_vec = FortranArray(np.zeros(6), dtype=np.float64)
        r_vec[:] = [1, 2, 3, 4, 5, 6]
        self.assertIs(type(r_vec), FortranArray)
        self.assertEqual(r_vec.dtype, np.float64)
        self.assertTrue(np.allclose(r_vec, [1, 2, 3, 4, 5, 6]))
        r_vec[::-1] = [1, 2, 3, 4, 5, 6]
        self.assertIs(type(r_vec), FortranArray)
        self.assertEqual(r_vec.dtype, np.float64)
        self.assertTrue(np.allclose(r_vec, [6, 5, 4, 3, 2, 1]))

        # Partial slice.
        c_vec = FortranArray([[1], [1], [2], [3], [5], [8], [13], [21]])
        self.assertEqual(c_vec.dtype, np.int32)
        c_vec[3:6, :] = c_vec[6:3:-1, :]  # Swap central values.
        self.assertTrue(np.all(c_vec.T == [1, 1, 8, 5, 3, 2, 13, 21]))

        # Swap off-axis array blocks in one step.
        fib_ref = ((1, 1, 2, 3),
                   (5, 8, 13, 21),
                   (34, 55, 89, 144),
                   (233, 377, 610, 987))
        fib_mat = FortranArray(fib_ref)
        fib_mat[1:2, 3:4], fib_mat[3:4, 1:2] = (fib_mat[3:4, 1:2].copy(),
                                                fib_mat[1:2, 3:4].copy())
        self.assertEqual(fib_mat.dtype, np.int32)
        self.assertTrue(np.all(fib_mat[1:2, 1:2] ==  # Top left.
                               np.asarray(fib_ref)[0:2, 0:2]))
        self.assertTrue(np.all(fib_mat[1:2, 3:4] ==  # Top right (swapped).
                               np.asarray(fib_ref)[2:4, 0:2]))
        self.assertTrue(np.all(fib_mat[3:4, 1:2] ==  # Bottom left (swapped).
                               np.asarray(fib_ref)[0:2, 2:4]))
        self.assertTrue(np.all(fib_mat[3:4, 3:4] ==  # Bottom right.
                               np.asarray(fib_ref)[2:4, 2:4]))

    def test_arithmetic(self):
        from pyavia import FortranArray
        import numpy as np

        # Test inversion, multiplication.
        a_mat = FortranArray([[1, 2, 3], [0, 1, 4], [5, 6, 0]],
                             dtype=np.float64)
        a_inv = np.linalg.inv(a_mat)
        prod_mat = a_mat @ a_inv

        self.assertIs(type(a_inv), FortranArray)
        self.assertIs(type(prod_mat), FortranArray)
        self.assertTrue(np.allclose(prod_mat, np.eye(3)))

        r_vec = FortranArray([[1, 1, 1, 1]], dtype=np.float64)
        c_vec = r_vec.T
        all_one = c_vec @ r_vec
        one_one = r_vec @ c_vec
        self.assertEqual(all_one.shape, (4, 4))
        self.assertTrue(np.allclose(all_one, 1.0))
        self.assertEqual(one_one.shape, (1, 1))
        self.assertAlmostEqual(one_one[1, 1], 4.0)

        # Subtraction.
        sub_mat = a_mat - a_mat
        self.assertIs(type(sub_mat), FortranArray)
        self.assertTrue(np.allclose(sub_mat, 0.0))

        # Negation.
        a_mat_neg = -a_mat
        self.assertTrue(np.allclose(a_mat + a_mat_neg, 0.0))


class TestWeightedDirGraph(TestCase):
    def test___init__(self):
        from pyavia import WtDirgraph, g_link
        wdg = WtDirgraph()

        # Test basic functions.
        wdg['a':'b'] = 'somevalue'
        self.assertIn('a', wdg)  # Both keys created by link assignment.
        self.assertIn('b', wdg)
        self.assertEqual(wdg['a':'b'], 'somevalue')
        with self.assertRaises(KeyError):
            print(wdg['b':'a'])  # Reverse should not exist.

        # Test reverse link.
        wdg['b':'a'] = 1.23
        self.assertEqual(wdg['b':'a'], 1.23)
        self.assertNotEqual(wdg['a':'b'], wdg['b':'a'])

        # Test heterogeneous and multiple keys.
        wdg['a':3.14159] = (22, 7)
        wdg[456:True] = 'Yes'
        with self.assertRaises(KeyError):
            wdg[1:2:3] = 4  # Invalid kind of slice index.
        self.assertNotEqual(wdg['a':'b'], wdg['a':3.14159])

        # Test key deletion and contains.
        del wdg['a':'b']  # Specific x -> y
        self.assertNotIn(g_link('a', 'b'), wdg)
        self.assertIn(g_link('b', 'a'), wdg)  # Reverse should not be deleted.
        del wdg[456]  # Entire x-key.
        with self.assertRaises(KeyError):
            del wdg[3.14159, 'a']  # Reverse should not exist.
            del wdg[456, True]  # Should already be gone.

        # Can't set path to nowhere.
        with self.assertRaises(KeyError):
            wdg['a':'a'] = 666

        # Test construction with forwards dict.
        wdg = WtDirgraph({'a': {'b': 2, 'c': 5}, 'c': {'a': 4}})
        self.assertEqual(wdg['c':'a'], 4)
        with self.assertRaises(KeyError):
            print(wdg['b':'a'])

    def test_trace(self):
        from pyavia import WtDirgraph
        wdg = WtDirgraph()
        wdg[1:2] = 0.5
        wdg[1:3] = 0.2
        wdg[1:4] = 5
        wdg[2:7] = 1
        wdg[2:8] = 3.14159
        wdg[7:-1] = -2

        # Simple paths should be lists with two nodes.
        self.assertEqual(wdg.trace(2, 7), [2, 7])

        # Path to nowhere is invalid.
        with self.assertRaises(KeyError):
            wdg.trace(4, 4)

        # Even simple paths should not be reversible.
        self.assertEqual(wdg.trace(7, 2), None)

        # Check complex forward path.
        path, path_sum = wdg.trace(1, -1, op=lambda x, y: x + y)
        self.assertEqual(path, [1, 2, 7, -1])
        self.assertEqual(path_sum, -0.5)

        # Forward path check (#2 check side-effects of caching).
        path, path_sum = wdg.trace(1, -1, op=lambda x, y: x + y)
        self.assertEqual(path, [1, 2, 7, -1])
        self.assertEqual(path_sum, -0.5)

        # No reverse path should exist.
        path, path_sum = wdg.trace(-1, 1, op=lambda x, y: x + y)
        self.assertIsNone(path)
        self.assertIsNone(path_sum)

        # Add reverse path and confirm it now exists and is different.
        wdg[-1:3] = 5
        wdg[3:1] = 7
        path, path_sum = wdg.trace(-1, 1, op=lambda x, y: x + y)
        self.assertEqual(path, [-1, 3, 1])
        self.assertEqual(path_sum, 12)

        # Forward path check (#3 check side-effects of caching reverse).
        path, path_sum = wdg.trace(1, -1, op=lambda x, y: x + y)
        self.assertEqual(path, [1, 2, 7, -1])
        self.assertEqual(path_sum, -0.5)

        # Reverse path check (#2 check side-effects of caching and fwd path).
        path, path_sum = wdg.trace(-1, 1, op=lambda x, y: x + y)
        self.assertEqual(path, [-1, 3, 1])
        self.assertEqual(path_sum, 12)
