from typing import Any

import numpy as np
from numpy import ndarray

try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


class Step(BaseAct):
    """Step-Activation function class.

    Examples:
    >>> import numpy as np
    >>> from main.dl.act._plot import plot_graph
    >>> print(Step())
    Step()
    >>> x = np.arange(-10, 10, 5e-2)
    >>> plot_graph(x, "step")
    """

    def forward(self, x: ndarray, *args: Any, **kwds: Any) -> ndarray:
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'Step' function.
        """
        return np.where(x>0, 1, 0)

    def backward(self, x: ndarray, *args: ndarray, **kwds: Any) -> ndarray:
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): An array of the same shape as the input,
                         with all zeros.
        """
        return np.zeros_like(x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
