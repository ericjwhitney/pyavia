"""
Small, general purpose utility functions.
"""

# Written by: Eric J. Whitney  Last updated: 21 January 2022.


import uuid


# == File Functions ===========================================================


def temp_filename(prefix: str = '', suffix: str = '', rand_length: int = None):
    """
    Generates a (nearly unique) temporary file name with given `prefix` and
    `suffix` using a hex UUID, truncated to `rand_length` if required.  This
    is useful for interfacing with older DOS and FORTRAN style codes which
    may have specific rules about filename length.
    """
    return prefix + uuid.uuid4().hex[:rand_length] + suffix
