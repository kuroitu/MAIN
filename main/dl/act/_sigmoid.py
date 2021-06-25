from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class Sigmoid(BaseAct):
    """Sigmoid-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(Sigmoid())
    Sigmoid()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "sigmoid")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Step' function.
        """
        return 1/(1+np.exp(-x))

    def backward(self, x: ndarray, y: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.
            y (ndarray): Outpu array after activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return y*(1-y)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
