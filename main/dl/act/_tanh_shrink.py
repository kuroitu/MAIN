from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class TanhShrink(BaseAct):
    """Tanh-Shrink-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(TanhShrink())
    TanhShrink()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "tanhshrink")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'TanhShrink' function.
        """
        return x-np.tanh(x)

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return np.tanh(x)**2


if __name__ == "__main__":
    import doctest
    doctest.testmod()
