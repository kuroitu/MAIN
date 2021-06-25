from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class ReLU6(BaseAct):
    """ReLU6-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(ReLU6())
    ReLU6()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "relu6")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'ReLU6' function.
        """
        return np.clip(x, 0, 6)

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return np.where((0<x)&(x<6), 1, 0)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
