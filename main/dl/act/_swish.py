from typing import Any
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


@dataclass
class Swish(BaseAct):
    """Swish-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(Swish())
    Swish(beta=1)
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "swish")
    """
    beta: float = 1

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Swish' function.
        """
        return x/(1+np.exp(-self.beta*x))

    def backward(self, x: ndarray, y: ndarray, *args: Any, **kwds: Any) \
            -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.
            y (ndarray): Output array after activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return self.beta*y + (1 - self.beta*y)/(1+np.exp(-self.beta*x))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
