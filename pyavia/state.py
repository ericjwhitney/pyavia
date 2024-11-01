"""
===========================
State (:mod:`pyavia.state`)
===========================

.. currentmodule:: pyavia.state

XXXXX TODO
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Protocol


# Written by Eric J. Whitney, February 2023.

# ======================================================================

class InvalidStateError(ValueError):
    """
    Exception raised when an object is set to an invalid state.


    TODO THIS PART DELETE?
    .. note::Checks of state object property `valid_state` after this
       exception is raised should return `False` until the state is
       corrected.
    """
    pass


# ----------------------------------------------------------------------

class State(ABC):
    """
    Abstract base class for *stateful* objects in PyAvia.  These are
    designed to be persistent objects that give continuous and changing
    (i.e. 'live') outputs based on an operating condition specified by
    method `set_states`.

    The object is expected to be either be in a valid state at all times
    after construction, unless an `InvalidStateError` is raised by
    `__init__`, `set_state(...)` or another method.

    TODO REMOVE THIS BIT
    If the object is in
    an invalid state the property `valid_state` should return `False`
    until a further call to `set_state(...)` restores a valid state.
    """

    def __init__(self, *args, **kwargs):
        # Added to cover cases of odd multiple inheritance.
        if args:
            raise TypeError(f"Unexpected positional argument(s): "
                            f"{', '.join(args)}")

        if kwargs:
            kwarg_list = [f"{k}={v}" for k, v in kwargs.items()]
            raise TypeError(f"Unexpected keyword argument(s): "
                            f"{', '.join(kwarg_list)}")

    # -- Public Methods ------------------------------------------------

    # def all_states(self) -> frozenset[str]:
    #     """
    #     Returns the combined set from `input_states` and `output_states`.
    #     """
    #     return self.input_states() | self.output_states()

    # def def_states(self) -> frozenset[str]:
    #     """
    #     Returns one set of input state names (there may be many sets) that
    #     can be used to fully define the operating condition of the object,
    #     if passed to `set_states`.
    #
    #     Notes
    #     -----
    #     At the base level this method simply returns `input_states`,
    #     which assumes that all input states are independent.  If a derived
    #     type offers interrelated or redundant states, this method should be
    #     overriden to return a list of input states that uniquely defines the
    #     operating condition of the object and that would be accepted by
    #     `set_states`.
    #     """
    #     return self.input_states()

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Returns a collection of input attributes that can uniquely and
        completely define the object state if passed to `set_state`.
        Typically this will be the 'fundamental' or 'minimum set' of
        input states.



        Returns
        -------
        dict[str, Any]
            Dict containing each of the state names and its current
            value.

            .. note::At the base level this method returns an empty
               `dict`.

        Notes
        -----
        - Derived classes should override this method and chain up if
          inheritance is used.

        - Other combinations of input states may exist that can also
          define the object state.
        """
        return {}

    # @abstractmethod
    # def input_states(self) -> frozenset[str]:
    #     """
    #     Returns all input (i.e. writeable) state names. Any name in this set
    #     can be used as an argument to ``set_states(...)`` (although not
    #     necessarily at the same time).
    #
    #     Notes
    #     -----
    #     This method must be supplied by each `States` object.  If it is a
    #     derived type then typically it would chain up with its superclass
    #     in some way.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def output_states(self) -> frozenset[str]:
    #     """
    #     Returns output (i.e. read-only) states.  These states accnot be set
    #     using `set_states`.
    #
    #     Notes
    #     -----
    #     This method must be supplied by each `States` object.  If it is a
    #     derived type then typically it would chain up with its superclass
    #     in some way.
    #     """
    #     raise NotImplementedError

    def restore_state(self) -> RestoreState:
        """
        Returns a context manager object that restores the state of
        `self` to its original value after some interim operations are
        performed.

        Returns
        -------
        RestoreState
            Context manager object.
        """
        return RestoreState(self)

    @abstractmethod
    def set_state(self, **inputs) -> frozenset[str]:
        """
        Change the state of the object by setting the values of 
        particular input attributes.  Multiple inputs can be set 
        simultaneously.

        .. note::Derived classes should override this method and chain
           up if inheritance is used.

        Parameters
        ----------
        **inputs
            Keyword arguments consisting of ``state_name=value``.  All
            keywords must be valid input state names.

        Returns
        -------
        frozenset[str]
            Set of input state names whose values have changed.

            .. note::At the base level, `set_state(...)` returns an
               empty `frozenset`.

        Raises
        ------
        InvalidStateError
            If the supplied values leave object in an invalid state.

        Notes
        -----
        Even if an input state was provided in `**inputs`` it is not
        guaranteed to appear in the returned set:

        - If the value already matched the current state value the
          implementation may have avoided making any changes /
          recalculation.
        - The implementation may have elected to change a different
          state (e.g. a more 'funamental' state) to give the equivalent
          result.
        """
        return frozenset()

    # @property
    # @abstractmethod
    # def valid_state(self) -> bool:
    #     """
    #     Returns `True` if the object is considered to be in a `valid`
    #     state, i.e. the input states are such that all output states
    #     can be correctly computed.  If not, this returns `False`.
    #     """
    #     raise NotImplementedError


# ----------------------------------------------------------------------

class RestoreState:
    """
    Context manager object that restores the state of `obj` to its
    original value after some interim operations are performed.

    Parameters
    ----------
    obj : State
        The object for which the states need to be restored.
    """
    def __init__(self, obj: State):
        self.obj = obj

    def __enter__(self):
        self.restore_point = self.obj.get_state()
        return self.obj

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.obj.set_state(**self.restore_point)

# ======================================================================


class RestoreStateProto:
    """
    Context manager object that restores the state of `obj` to its
    original value after some interim operations are performed.

    Parameters
    ----------
    obj : StateProto
        The object for which the states need to be restored.
    """
    def __init__(self, obj: StateProto):
        self.obj = obj

    def __enter__(self):
        self.restore_point = self.obj.get_state()
        return self.obj

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.obj.set_state(**self.restore_point)

# ----------------------------------------------------------------------


# class StateProto(Protocol):
#     """
#     Protocol for *stateful* objects in PyAvia.  These are designed to be
#     'living' objects that give continuous and changing outputs based on
#     an operating condition specified by method `set_states`.
#
#     The object is expected to be in a valid state at all times after
#     construction, unless an `InvalidStateError` is raised by `__init__`,
#     `set_state(...)` or another method.  If an `InvalidStateError`
#     exception is raised the user is expected to handle the issue, where
#     further call/s to `set_state(...)` no longer raise an exception.
#     """
#
#     # -- Public Methods ------------------------------------------------
#
#     def get_state(self) -> dict[str, Any]:
#         """
#         Returns a collection of input attributes that can uniquely and
#         completely define the object state if passed to `set_state`.
#         Typically this will be the 'fundamental' or 'minimum set' of
#         input states.
#
#         .. note:: Other combinations of input states may exist that can
#            also define the object state.
#
#         Returns
#         -------
#         dict[str, Any]
#             Dict containing each of the state names and its current
#             value.
#         """
#         ...
#
#     def restore_state(self) -> RestoreStateProto:
#         """
#         Returns a context manager object that restores the state of
#         `self` to its original value after some interim operations are
#         performed.
#
#         Returns
#         -------
#         RestoreState
#             Context manager object.
#         """
#         return RestoreStateProto(self)
#
#     def set_state(self, **inputs) -> frozenset[str]:
#         """
#         Change the state of the object by setting the values of
#         particular input attributes.  Multiple inputs can be set
#         simultaneously.  It must be supplied by a derived class.
#
#         Parameters
#         ----------
#         **inputs
#             Keyword arguments consisting of ``state_name=value``.  All
#             keywords must be valid input state names.
#
#         Returns
#         -------
#         frozenset[str]
#             Set of input state names whose values have changed.
#
#         Raises
#         ------
#         InvalidStateError
#             If the supplied values leave object in an invalid state.
#
#         Notes
#         -----
#         Even if an input state was provided in `**inputs`` it is not
#         guaranteed to appear in the returned set:
#
#         - If the value already matched the current state value the
#           implementation may have avoided making any changes /
#           recalculation.
#         - The implementation may have elected to change a different
#           state (e.g. a more 'funamental' state) to give the equivalent
#           result.
#         """
#         ...
