from abc import ABC, abstractmethod
from typing import Any


# Written by Eric J. Whitney, February 2023.


# ======================================================================
class InvalidStateError(ValueError):
    """
    Exception raised when an object is set to an invalid state.
    """
    # TODO remove additional: Checks of `obj.valid_state` property after
    #  this point should return `False` until the state is corrected.
    pass


# ----------------------------------------------------------------------

class States(ABC):
    """
    Abstract base class for *stateful* objects in PyAvia.  These are
    designed to be persistent objects that give continuous and changing
    (i.e. 'live') outputs based on an operating condition specified by
    method `set_states`.  States are assumed to be available from the
    time of initialisation onwards.
    """

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
    def get_states(self) -> dict[str, Any]:
        """
        Subclasses should override this method to returns a dict of
        input states that can uniquely and completely define the object
        state if passed to `set_states`.  Typically this will be the
        'fundamental' or 'minimum set' of input states.

        .. note:: Other combinations of input states may exist that can
           also define the object state.

        Returns
        -------
        dict[str, Any]
            Dict containing each of the state names and its current
            value.
        """
        raise NotImplementedError

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

    def restore_states(self) -> 'RestoreStates':
        return RestoreStates(self)

    @abstractmethod
    def set_states(self, **states) -> frozenset[str]:
        """
        Set input state/s of the object.  This method allows for
        setting multiple states simultaneously.  It must be supplied by
        a derived class.

        Parameters
        ----------
        states : dict[str, Any]
            Keyword arguments consisting of ``state_name=value``.  All
            keywords must be valid input state names.

        Returns
        -------
        frozenset[str]
            Set of input state names whose values have changed.

        Raises
        ------
        InvalidStateError
            If the supplied values leave object in an invalid state.
            The property `valid_state` will return `False` until the
            state is corrected.

        Notes
        -----
        Even if an input state was provided in `**states`` it is not
        guaranteed to appear in the returned set:

        - If the value already matched the current state value the
          implementation may have avoided making any changes /
          recalculation.
        - The implementation may have elected to change a different
          state (e.g. a more 'funamental' state) to give the equivalent
          result.
        """
        raise NotImplementedError

    # TODO Superfluous, just rely on exceptions?
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

class RestoreStates:
    """
    Context manager object that restores the states of `obj` to their
    original value after some interim operations are performed.

    Parameters
    ----------
    obj : States
        The object for which the states need to be restored.

    """
    def __init__(self, obj: States):
        self.obj = obj

    def __enter__(self):
        self.restore_point = self.obj.get_states()

        return self.obj

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.obj.set_states(**self.restore_point)

# ======================================================================
