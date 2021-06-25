from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseError
except ImportError:
    # For doctest
    from main.dl.err import BaseError


class Square(BaseError):
    """Square-Error function class.

    Examples:
    >>> import numpy as np
    >>> obj = Square()
    >>> print(obj)
    Square(error=0)
    >>> obj.forward(np.array([0, 0.5, 1]), np.array([0, 1, 0]))
    array([0.   , 0.125, 0.5  ])
    >>> obj.backward(np.array([0, 0.5, 1]), np.array([0, 1, 0]))
    array([ 0. , -0.5,  1. ])
    """

    def forward(self, y: ndarray, t: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Forward propagation.

        Args:
            y (ndarray): Inference result.
            t (ndarray): Correct label.

        Returns:
            _ (ndarray): Error between correct answer and inference.
        """
        return 0.5*(y-t)**2

    def backward(self, y: ndarray, t: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Backward propatation.

        Args:
            y (ndarray): Inference result.
            t (ndarray): Correct label.

        Returns:
            _ (ndarray): Partial derivative of y with respect to t.
        """
        return y-t


if __name__ == "__main__":
    import doctest
    doctest.testmod()
