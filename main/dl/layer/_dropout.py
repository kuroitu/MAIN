from dataclasses import dataclass

import numpy as np

try:
    from ._base import BaseLayer
except:
    # For doctest
    from main.dl.layer import BaseLayer


@dataclass
class Dropout(BaseLayer):
    """Dropout layer class.
    In this class implemented 'Reverse Dropout'.
    'Reverse Dropout';
        When training, increase the density of the output.
        When inference, leave the density of the output as it is.
    If you want to do 'Normal Dropout' then specify 'name' = "ndrop".

    Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> prev = 10
    >>> obj = Dropout(prev=prev)
    >>> print(obj)
    Dropout(prev=10, n=10, ratio=0.25)
    >>> x = np.linspace(0, 1, prev)
    >>> y = obj.forward(x)
    >>> print(y)
    [[0.         0.14814815 0.2962963  0.44444444 0.59259259 0.
      0.88888889 0.         0.         1.33333333]]
    >>> grad = np.linspace(0, 1, prev)
    >>> grad_x = obj.backward(grad)
    >>> print(grad_x)
    [[0.         0.14814815 0.2962963  0.44444444 0.59259259 0.
      0.88888889 0.         0.         1.33333333]]
    >>> obj = Dropout(name="ndrop", prev=prev)
    >>> print(obj)
    Dropout(prev=10, n=10, ratio=0.25)
    >>> y = obj.forward(x)
    >>> print(y)
    [[0.         0.11111111 0.22222222 0.         0.44444444 0.55555556
      0.66666667 0.         0.88888889 0.        ]]
    >>> grad_x = obj.backward(grad)
    >>> print(grad_x)
    [[0.         0.11111111 0.22222222 0.         0.44444444 0.55555556
      0.66666667 0.         0.88888889 0.        ]]
    """
    ratio: float = 0.25

    def __post_init__(self, *args, **kwds):
        super().__post_init__(*args, **kwds)

        self.mask = np.empty((1, self.prev))
        self.n = self.prev
        if "ndrop" in self.name:
            self.forward = self._normal_forward
            self.backward = self._normal_backward

    def forward(self, x, *args, train=True, **kwds):
        """Forward propagation.

        Args:
            x (ndarray): Input array.
            train (bool): Training or not flag.

        Returns:
            _ (ndarray): Output array after masking.
        """
        if train:
            self.mask = np.random.randn(*self.mask.shape)
            self.mask = np.where(self.mask>=self.ratio, 1, 0)
            return x*self.mask/(1-self.ratio)
        else:
            return x

    def backward(self, grad, *args, **kwds):
        """Backward propagation.

        Args:
            grad (ndarray): Gradient flowing from the lower layer.

        Returns:
            _ (ndarray): Gradient flowing to the upper layer.
        """
        return grad*self.mask/(1-self.ratio)

    def _normal_forward(self, x, *args, train=True, **kwds):
        """For 'Normal Dropout' forward propagation."""
        if train:
            self.mask = np.random.randn(*self.mask.shape)
            self.mask = np.where(self.mask>=self.ratio, 1, 0)
            return x*self.mask
        else:
            return x/(1-self.ratio)

    def _normal_backward(self, grad, *args, **kwds):
        """For 'Normal Dropout' backward propagation."""
        return grad*self.mask

    def update(self, *args, **kwds):
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
