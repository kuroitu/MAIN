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
class Cross(BaseError):
    """Cross-Error function class.

    Examples:
    >>> import numpy as np
    >>> obj = Cross()
    >>> print(obj)
    Cross(error=0, eps=1e-08)
    >>> obj.forward(np.array([0, 0.5, 1]), np.array([0, 1, 0]))
    array([-0.        ,  0.69314716,  0.        ])
    >>> obj.backward(np.array([0, 0.5, 1]), np.array([0, 1, 0]))
    array([ 0.        , -1.99999996,  0.        ])
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
        return -t*np.log(y+self.eps)

    def backward(self, y: ndarray, t: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Backward propatation.

        Args:
            y (ndarray): Inference result.
            t (ndarray): Correct label.

        Returns:
            _ (ndarray): Partial derivative of y with respect to t.
        """
        return -t/(y+self.eps)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
