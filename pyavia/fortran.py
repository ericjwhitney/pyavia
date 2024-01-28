"""
Algorithms and containers mimicking Fortran-style behaviour.  This allows
interoperability with - or straightforward implementation of - algorithms
originally written using Fortran or MATLAB.

Examples
--------
Create a 4x4 array using Fibonacci numbers:

    >>> fib_mat = fortran_array([[1, 1, 2, 3],
    ...                         [5, 8, 13, 21],
    ...                         [34, 55, 89, 144],
    ...                         [233, 377, 610, 987]])
    >>> print(repr(fib_mat))
    fortran_array([[  1,  1,  2,  3],
                   [  5,  8, 13, 21],
                   [ 34, 55, 89,144],
                   [233,377,610,987]])

Swap the off-diagonal blocks using Fortran indices (copy prevents
overwrite):

    >>> fib_mat[1:2, 3:4], fib_mat[3:4, 1:2] = (fib_mat[3:4, 1:2].copy(),
    ...                                         fib_mat[1:2, 3:4].copy())
    >>> print(repr(fib_mat))
    fortran_array([[  1,  1, 34, 55],
                   [  5,  8,233,377],
                   [  2,  3, 89,144],
                   [ 13, 21,610,987]])

Invert just the bottom left 3x3 using NumPy.  Note that this also
returns a ``FortranArray``:

    >>> inv_fib = np.linalg.inv(fib_mat[2:4, 1:3])
    >>> print(repr(inv_fib))
    fortran_array([[ 4.57396837e+14,-1.52465612e+14,-1.52465612e+14],
                   [ 7.38871814e+14,-2.46290605e+14,-2.46290605e+14],
                   [-3.51843721e+13, 1.17281240e+13, 1.17281240e+13]])

As another example, taking a given definite matrix, show that multiplication
by its inverse gives the unit matrix:

        >>> p = fortran_array([[1, -1,  2,  0],
        ...                   [-1,  4, -1,  1],
        ...                   [2,  -1,  6, -2],
        ...                   [0,  1,  -2,  4]], ftype='real*8')
        >>> p_inv = np.linalg.inv(p)
        >>> print(p_inv @ p)
        [[ 1.00000000e+00,-3.55271368e-15, 0.00000000e+00, 0.00000000e+00],
         [ 8.88178420e-16, 1.00000000e+00, 1.77635684e-15,-8.88178420e-16],
         [-1.77635684e-15, 3.10862447e-15, 1.00000000e+00, 1.77635684e-15],
         [-8.88178420e-16, 1.33226763e-15,-2.66453526e-15, 1.00000000e+00]]
"""

# Last updated: 8 March 2021 by Eric J. Whitney

from __future__ import annotations
import numpy as np

from pyavia.containers import MultiBiDict

__all__ = ['fortran_do', 'fortran_array', 'FortranArray']


# -----------------------------------------------------------------------------

# noinspection PyTypeChecker
def fortran_do(start: int, stop: int, step: int = 1) -> range:
    """Returns sequence in a similar style to ``range()``, but gives the same
    values as the Fortran style do-loop construct.  Differences are:

        - The range is inclusive, i.e. ``start <= i <= stop`` or
          ``start >= i >= stop``.
        - Start and stop are both required (default step = 1 remains).

    Examples
    --------
    Normal fortran sequences are inclusive of the end value:

        >>> list(fortran_do(1, 10))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    As a more complete example, we can directly implement a Fortran-style
    algorithm in Python.  First we declare the matricies and arrays:

        >>> a = FortranArray(4, 4, ftype='real*8')
        >>> b = FortranArray(4, ftype='integer')

    Assign values using Fortran indices:

        >>> for i in fortran_do(1, 4):
        ...     b[i] = i * 2
        ...     for j in fortran_do(1, 4):
        ...         a[i, j] = i + j

    This gives the values:

        >>> print(a)
        [[2.,3.,4.,5.],
         [3.,4.,5.,6.],
         [4.,5.,6.,7.],
         [5.,6.,7.,8.]]

        >>> print(b)
        [2,4,6,8]
    """
    return range(start, stop + (1 if step > 0 else -1), step)


# -----------------------------------------------------------------------------

# Mapping between Fortran and Numpy types.
_F2NP_TYPES = MultiBiDict({
    'logical*1': 'bool_',
    'logical': 'bool_',  # LOGICAL Default.
    'complex*32': 'complex64',  # Fortran specifies size of each part.
    'complex*64': 'complex128',
    'complex': 'complex128',  # COMPLEX Default.
    'integer*1': 'int8',
    'integer*2': 'int16',
    'integer*4': 'int32',
    'integer*8': 'int64',
    'integer': 'int32',  # INTEGER Default.
    'real*4': 'float32',
    'real*8': 'float64',
    'real': 'float64'  # REAL Default.
    # Defaults of each type are placed last so that .ftype() reports the
    # longer version first.
})


def fortran_array(arr, dtype=None, *, copy=True, order='F', subok=False,
                  ndmin=1, ftype=None):
    """
    Return a new FortranArray object using identical arguments to
    `np.array()`, with the following differences:

        - Fortran type `ftype` (case insensitive) may be given instead of
          `dtype` (but not both).  See `FortranArray.__new__` for types.
        - Default layout is ``order='F'``.
        - Minimum of ``ndmin=1``  is enforced.
        - `like` keyword is not provided.
    """
    if ndmin < 1:
        raise ValueError("FortranArrays must be of dimension one or higher.")
    if ftype and dtype:
        raise AttributeError("Cannot specify both Numpy and Fortran types.")

    if ftype:
        try:
            dtype = _F2NP_TYPES[ftype.casefold()]
        except KeyError:
            raise ValueError(f"Fortran type {ftype} not supported.")

    return np.array(arr, dtype=dtype, copy=copy, order=order, subok=subok,
                    ndmin=ndmin).view(FortranArray)


# -----------------------------------------------------------------------------

class FortranArray(np.ndarray):
    """
    Numpy array subclass emulating a Fortran-style index-1 array that can
    be used directly when implementing algorithms from Fortran or Matlab.
    Indexing supports slice-off-the-end which is valid in Fortran 90 and
    NumPy.

    Behaviour matches `np.ndarray` except:
        - There is no element [0], [0, 0], ... as expected.
        - Negative indexing to access end elements is not allowed.
        - NumPy advanced indexing is not allowed.
        - The additional `FortranArray.ftype` property returns the Fortran
          equivalent of the underlying Numpy data type in use.

    .. Note:: `FortranArray` values by default are initialised to zero,
              however this can be changed by setting
              `FortranArray.INIT_DEFAULT`.  This is because different Fortran
              compilers offer different system-wide initialisation policies
              such as zero, nothing / garbage, etc.
    """

    INIT_DEFAULT = 0  # Init. value for directly constructed FortranArray().

    def __new__(cls, *dims, ftype='real*8'):
        """
        Creates a new `FortranArray`.

        Parameters
        ----------
        dims : int, int, ...
            Size of the array on each axis.  At least one dimension is
            required (zero dimension FortranArrays are not supported).

        ftype : str (case insensitive)
            Fortran datatype for array (default = 'real*8').  This is
            converted to a corresponding Numpy type:

                - 'logical*1':  `bool_`
                - 'logical':    `bool_`       (default logical)
                - 'complex*32': `complex64`
                - 'complex*64': `complex128`
                - 'complex':    `complex128`  (default complex)
                - 'integer*1':  `int8`
                - 'integer*2':  `int16`
                - 'integer*4':  `int32`
                - 'integer*8':  `int64`
                - 'integer':    `int32`       (default integer)
                - 'real*4':     `float32`
                - 'real*8':     `float64`
                - 'real':       `float64`     (default floating point)

        Returns
        -------
        result : FortranArray
            Subclass of `np.ndarray` of type `ftype`, using Fortran ordering,
            with all values initialised to `INIT_DEFAULT`.

        Raises
        ------
        AttributeError
            If `*dims` are not provided.
        ValueError
            If `ftype` is invalid.

        Examples
        --------
        Create a 4x4 array of 64-bit floating point values.

            >>> arr = FortranArray(4, 4, ftype='REAL*8')
            >>> print(arr.shape)
            (4, 4)
            >>> print(arr.dtype)
            float64
        """
        if not dims:
            raise AttributeError("At least one array dimension must be "
                                 "provided.")
        try:
            np_type = _F2NP_TYPES[ftype.casefold()]
        except KeyError:
            raise ValueError(f"Fortran type '{ftype}' not supported.")

        return np.full(shape=dims, fill_value=cls.INIT_DEFAULT,
                       dtype=np.dtype(np_type), order='F').view(cls)

    def __getitem__(self, key):
        """
        Get array element using Fortran-style indexing.
        """
        return np.ndarray.__getitem__(self, self._f2py_idx(key)).view(
            FortranArray)

    def __setitem__(self, key, value):
        """
        Set array element using Fortran-style indexing to `value`.
        """
        self.view(np.ndarray)[self._f2py_idx(key)] = np.asarray(value)

    def __repr__(self):
        prefix, suffix = 'fortran_array(', ')'
        return (prefix +
                np.array2string(self.view(np.ndarray), separator=',',
                                prefix=prefix, suffix=suffix) +
                suffix)

    def __str__(self):
        return np.array2string(self.view(np.ndarray), separator=',')

    @property
    def ftype(self):
        """
        Returns a string giving the first Fortran type matching the internal
        Numpy dtype used.
        """
        return _F2NP_TYPES.inverse[str(self.dtype)][0]

    # -- Private Methods ----------------------------------------------------

    def _f2py_idx(self, fort_idx):
        """
        Convert Fortran-style array slices (index-1) or individual indices
        into Python / C equivalent (index-0).
        """
        if not isinstance(fort_idx, tuple):
            fort_idx = fort_idx,  # Handle 0-D requests.
        if len(fort_idx) != self.ndim:
            raise TypeError(f"Incorrect number of indices: Got "
                            f"{len(fort_idx)}, needed {self.ndim}.")
        py_idx = []
        for ax_idx, ax_dim in zip(fort_idx, self.shape):
            if isinstance(ax_idx, slice):
                py_step = ax_idx.step or 1
                if py_step == 0:
                    raise IndexError("Zero stride not allowed.")

                if py_step >= 0:
                    py_start = (ax_idx.start or 1) - 1
                    py_stop = ax_idx.stop or ax_dim
                else:
                    py_start = (ax_idx.start or ax_dim) - 1
                    py_stop = (ax_idx.stop or 1) - 2
                    if py_stop < 0:
                        py_stop = None

                if 0 <= py_start and (py_stop is None or 0 <= py_stop):
                    py_idx.append(slice(py_start, py_stop, py_step))
                else:
                    raise IndexError(f"Invalid Fortran-style array slice: "
                                     f"{ax_idx}")

            else:
                if not 1 <= ax_idx <= ax_dim:
                    raise IndexError(f"Fortran-style index out of range:"
                                     f" {fort_idx}.")
                py_idx.append(ax_idx - 1)
        return tuple(py_idx)
