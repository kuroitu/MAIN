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
class SELU(BaseAct):
    """SELU-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(SELU())
    SELU(lambda_=1.0507, alpha=1.67326)
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "selu")
    """
    lambda_: float = 1.0507
    alpha: float = 1.67326

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'SELU' function.
        """
        return np.where(x>=0, self.lambda_*x,
                              self.lambda_*self.alpha*np.expm1(x))

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return np.where(x>=0, self.lambda_, self.lambda_*self.alpha*np.exp(x))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
