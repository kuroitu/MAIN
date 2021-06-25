from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class Softmin(BaseAct):
    """Softmin-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(Softmin())
    Softmin()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "softmin")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Softmin' function.
        """
        try:
            return np.exp(-x)/np.sum(np.exp(-x), axis=1, keepdims=True)
        except np.AxisError:
            return np.exp(-x)/np.sum(np.exp(-x))

    def backward(self, x: ndarray, y: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.
            y (ndarray): Output array after activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return -y*(1-y)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
