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
class ELU(BaseAct):
    """ELU-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(ELU())
    ELU(alpha=1)
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "elu")
    """
    alpha: float = 1

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'ELU' function.
        """
        return np.where(x>=0, x, self.alpha*np.expm1(x))

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return np.where(x>=0, 1, self.alpha*np.exp(x))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
