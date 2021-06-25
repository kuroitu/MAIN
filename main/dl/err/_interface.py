import string
from typing import Any

try:
    from ._base import BaseError
except ImportError:
    from main.dl.err import BaseError


def get_err(name: str, *args: Any, **kwds: Any) -> BaseError:
    """Interface for getting error function.

    Args:
        name (str): Target error function's name.

    Returns:
        _ (BaseError): Error function class.

    Example:
    >>> print(get_err("square"))
    Square(error=0)
    >>> print(get_err("binary"))
    Binary(error=0, eps=1e-08)
    >>> print(get_err("cross"))
    Cross(error=0, eps=1e-08)
    >>> print(get_err("hoge"))
    Traceback (most recent call last):
        ...
    KeyError: 'hoge'
    """
    from main.dl.err import err_dict

    name = name.lower().translate(str.maketrans("", "", string.punctuation))
    return err_dict[name](*args, **kwds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
