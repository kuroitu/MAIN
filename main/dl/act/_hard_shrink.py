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
class HardShrink(BaseAct):
    """Hard-Shrink-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(HardShrink())
    HardShrink(lambda_=0.5)
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "hardshrink")
    """
    lambda_: float = 0.5

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'HardShrink' function.
        """
        return np.where((-self.lambda_<=x)&(x<=self.lambda_), 0, x)

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        return np.where((-self.lambda_<=x)&(x<=self.lambda_), 0, 1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
