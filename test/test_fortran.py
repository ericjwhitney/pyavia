
from unittest.case import TestCase


class TestFortranArray(TestCase):
    # noinspection PyTypeChecker,PyUnusedLocal
    def test__init__(self):
        from pyavia.fortran import FortranArray, fortran_array
        import numpy as np

        # Test direct initialisation.
        a_mat = FortranArray(4, 4, ftype='real*8')
        self.assertEqual(a_mat.dtype, np.float64)
        self.assertEqual(a_mat.ftype, 'real*8')
        self.assertEqual(a_mat.shape, (4, 4))

        b_vec = FortranArray(4, ftype='integer')
        self.assertEqual(b_vec.dtype, np.int32)
        self.assertEqual(b_vec.ftype, 'integer*4')
        self.assertEqual(b_vec.shape, (4,))

        # Test initialisation from given data.
        c_mat = fortran_array([[3, 6], [-5, -9]], ftype='REAL*8')
        self.assertEqual(c_mat.dtype, np.float64)
        self.assertEqual(c_mat.shape, (2, 2))

        # Check invalid constructions.
        with self.assertRaises(AttributeError):
            dbl_types = fortran_array([3, 4], dtype=np.float64, ftype='real*8')
        with self.assertRaises(AttributeError):
            zero_mat = FortranArray(ftype='real')
        with self.assertRaises(ValueError):
            wrong_type = FortranArray(1, ftype='int')
        with self.assertRaises(TypeError):
            odd_args = FortranArray((2,))

    def test__getitem__(self):
        from pyavia.fortran import fortran_array
        import numpy as np

        # Test Fortran individual indexing.
        row_mat = fortran_array([1.0, 2.0, 3.0, 4.0])  # Row vector, float64.
        col_mat = fortran_array([[1], [2], [3], [4]])  # Column vector, int.
        full_mat = fortran_array([[3, 6], [-5, -9]], dtype=np.float64)

        self.assertEqual(row_mat[2], 2.0)
        self.assertEqual(row_mat[4].dtype, np.float64)
        self.assertEqual(row_mat.ftype, 'real*8')
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
        from pyavia.fortran import FortranArray, fortran_array
        import numpy as np

        # Test basic indices.
        a_vec = FortranArray(7)  # Default type is REAL.
        a_vec[1] = 4.0
        a_vec[7] = 4.5
        with self.assertRaises(IndexError):
            a_vec[0] = 3.2
        with self.assertRaises(IndexError):
            a_vec[8] = 5.0

        # Test slice indexing.
        r_vec = fortran_array(np.zeros(6), ftype='real')
        r_vec[:] = [1, 2, 3, 4, 5, 6]
        self.assertIs(type(r_vec), FortranArray)
        self.assertEqual(r_vec.dtype, np.float64)
        self.assertTrue(np.allclose(r_vec, [1, 2, 3, 4, 5, 6]))

        # Numpy and FortranArray slices work in the same positions.
        spl_vec_f = r_vec.copy()
        spl_vec_f[2:5] = fortran_array([11.0, 10.0, 9.0, 8.0])
        self.assertTrue(np.allclose(spl_vec_f, [1, 11, 10, 9, 8, 6]))
        spl_vec_n = r_vec.copy()
        spl_vec_n[2:5] = np.array([11.0, 10.0, 9.0, 8.0])
        self.assertTrue(np.allclose(spl_vec_f, spl_vec_n))

        # Reverse slice.
        r_vec[::-1] = [1, 2, 3, 4, 5, 6]
        self.assertIs(type(r_vec), FortranArray)
        self.assertEqual(r_vec.dtype, np.float64)
        self.assertTrue(np.allclose(r_vec, [6, 5, 4, 3, 2, 1]))

        # Partial slice.
        c_vec = fortran_array([[1], [1], [2], [3], [5], [8], [13], [21]])
        self.assertEqual(c_vec.dtype, np.int32)
        c_vec[3:6, :] = c_vec[6:3:-1, :]  # Swap central values.
        self.assertTrue(np.all(c_vec.T == [1, 1, 8, 5, 3, 2, 13, 21]))

        # Swap off-axis array blocks in one step.
        fib_ref = ((1, 1, 2, 3),
                   (5, 8, 13, 21),
                   (34, 55, 89, 144),
                   (233, 377, 610, 987))
        fib_mat = fortran_array(fib_ref)
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
        from pyavia.fortran import FortranArray, fortran_array
        import numpy as np

        # Test inversion, multiplication.
        a_mat = fortran_array([[1, 2, 3], [0, 1, 4], [5, 6, 0]],
                              ftype='real')
        a_inv = np.linalg.inv(a_mat)
        prod_mat = a_mat @ a_inv

        self.assertIs(type(a_inv), FortranArray)
        self.assertIs(type(prod_mat), FortranArray)
        self.assertTrue(np.allclose(prod_mat, np.eye(3)))

        r_vec = fortran_array([[1, 1, 1, 1]], dtype=np.float64)
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

    def test_arrayops(self):
        from pyavia.fortran import fortran_array
        import numpy as np

        fib_mat = fortran_array([[1, 1, 2, 3],
                                 [5, 8, 13, 21],
                                 [34, 55, 89, 144],
                                 [233, 377, 610, 987]], ftype='REAL')

        # Sum (a typical operaiton on all elements).
        self.assertAlmostEqual(fib_mat.sum(), 2583)
        self.assertTrue(np.allclose(fib_mat.sum(axis=0), [273, 441, 714, 1155]))
        self.assertTrue(np.allclose(fib_mat.sum(axis=1), [7, 47, 322, 2207]))
        self.assertEqual(fib_mat.flatten().shape, (16,))

        # Matrix inverse.
        p = fortran_array([[1, -1,  2,  0],
                           [-1, 4, -1,  1],
                           [2, -1,  6, -2],
                           [0,  1, -2,  4]], ftype='real*8')
        p_inv = np.linalg.inv(p)
        self.assertTrue(np.allclose(p @ p_inv, np.eye(4, dtype=np.float64)))
        self.assertTrue(np.allclose(p_inv @ p, np.eye(4, dtype=np.float64)))

    def test_typical_example(self):
        from pyavia.fortran import FortranArray, fortran_do

        a = FortranArray(4, 4, ftype='real*8')
        b = FortranArray(4, ftype='integer')
        for i in fortran_do(1, 4):
            b[i] = i * 2
            for j in fortran_do(1, 4):
                a[i, j] = i + j
