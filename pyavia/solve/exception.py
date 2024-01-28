

# Written by Eric J. Whitney, April 2023.


# ======================================================================

class SolverError(RuntimeError):
    """
    This exception is raised when an algorithm / solver / etc fails to
    converge or find a solution.  Additional information (optional) is
    included to allow the reason for the failure to be determined.

    Notes
    -----
    `SolverError` may also have additional attributes not listed here
    depending on the specific solver being used.
    """

    def __init__(self, *args, flag: int = None, details: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        args :
            Passed to `RuntimeError`.
        flag : int, default = None
            Numeric status code giving some information about the
            result. Typically `flag` != 0 as many error code systems
            assume that `flag` == 0 implies that the solution was
            successful.
        details : str, default = None
            Additional text can be included relating to the specific
            type of failure.
        kwargs :
            Additional attributes can be added to the object using
            keyword arguments.
        """
        super().__init__(*args)
        self.flag, self.details = flag, details
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        """Add additional details below the main failure notice."""
        error_str = super().__str__()
        for k, v in self.__dict__.items():
            if v is not None:
                error_str += f"\n{k} -> {v}"
        return error_str
