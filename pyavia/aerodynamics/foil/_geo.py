from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from pyavia.util.type_ops import make_sentinel

_MISSING = make_sentinel()

# TODO MOVE THIS FILE

# ===========================================================================


class FoilGeo(ABC):
    """
    Class defining the basic geometry model common to all aerofoils.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -- Public Methods -----------------------------------------------------

    @abstractmethod
    def xz(self, *, t: ArrayLike = _MISSING) -> ArrayLike:
        """
        xxx
        - No arguments:  Return the list of points that define the foil
          perimeter internally.
        - `t` xxxx

        TODO More... define t
        Parameters
        ----------
        t

        Returns
        -------
        pts : shape (2, n), array_like
            (`x`, `z`) pairs of points along the aerofoil perimeter.

        """
        raise NotImplementedError

    # -- Private Methods ----------------------------------------------------
