from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class Mish(BaseAct):
    """Mish-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(Mish())
    Mish()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "mish")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Mish' function.
        """
        return x*np.tanh(np.logaddexp(x, 0))

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return (np.exp(x)
                *(4*(x+1) + 4*np.exp(2*x) + np.exp(3*x) + (4*x + 6)*np.exp(x))
                /(2*np.exp(x) + np.exp(2*x) + 2)**2)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
