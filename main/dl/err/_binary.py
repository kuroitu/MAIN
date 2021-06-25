from typing import Any
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseError
except ImportError:
    # For doctest
    from main.dl.err import BaseError


@dataclass
class Binary(BaseError):
    """BinaryCross-Error function class.

    Examples:
    >>> import numpy as np
    >>> obj = Binary()
    >>> print(obj)
    Binary(error=0, eps=1e-08)
    >>> obj.forward(np.array([0, 0.5, 1]), np.array([0, 1, 0]))
    array([-9.99999989e-09,  6.93147161e-01,  1.84206807e+01])
    >>> obj.backward(np.array([0, 0.5, 1]), np.array([0, 1, 0]))
    array([ 0.00000000e+00, -1.99999992e+00,  1.00000000e+08])
    """
    eps: float = 1e-8

    def forward(self, y: ndarray, t: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Forward propagation.

        Args:
            y (ndarray): Inference result.
            t (ndarray): Correct label.

        Returns:
            _ (ndarray): Error between correct answer and inference.
        """
        return -t*np.log(y+self.eps) - (1-t)*np.log(1-y+self.eps)

    def backward(self, y: ndarray, t: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Backward propatation.

        Args:
            y (ndarray): Inference result.
            t (ndarray): Correct label.

        Returns:
            _ (ndarray): Partial derivative of y with respect to t.
        """
        return (y-t)/(y*(1-y) + self.eps)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
