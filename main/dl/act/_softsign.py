from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class Softsign(BaseAct):
    """Softsign-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(Softsign())
    Softsign()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "softsign")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Softsign' function.
        """
        return x/(1+np.abs(x))

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return 1/(1+np.abs(x))**2


if __name__ == "__main__":
    import doctest
    doctest.testmod()
