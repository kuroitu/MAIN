from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class HardSigmoid(BaseAct):
    """Hard-Sigmoid-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(HardSigmoid())
    HardSigmoid()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "hardsigmoid")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'HardSigmoid' function.
        """
        return np.clip(0.2*x + 0.5, 0, 1)

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return np.where((x>2.5)|(x<-2.5), 0, 0.2)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
