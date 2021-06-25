from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class Identity(BaseAct):
    """Identity-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from _plot import plot_graph
    >>> print(Identity())
    Identity()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "identity")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Identity' function.
        """
        return x

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): An array of the same shape as the input,
                         with all ones.
        """
        return np.ones_like(x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
