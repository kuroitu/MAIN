import sys
from typing import Union, Tuple
from dataclasses import dataclass, field

import numpy as np

from main.dl.util import im2col, col2im, get_Oshape
from main.dl.util._util import _get_shape, _get_stride, _get_pad
try:
    from ._base import BaseLayer
except ImportError:
    # For doctest
    from main.dl.layer import BaseLayer


@dataclass
class Pooling(BaseLayer):
    """Pooling layer class.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> prev, n = (4, 4), 2 # 'prev' means input array's shape, 'n' means padding window's size or stride width.
    >>> obj = Pooling(prev=prev, n=n)
    >>> print(obj)
    Pooling(prev=16, n=4, Ishape=(1, 1, 4, 4), Oshape=(1, 2, 2), stride=(2, 2), pad=(0, 0))
    >>> x = np.linspace(0, 1, obj.prev).reshape(*obj.Ishape)
    >>> y = obj.forward(x)
    >>> print(y)
    [[[[0.33333333 0.46666667]
       [0.86666667 1.        ]]]]
    >>> grad = np.linspace(0, 1, np.prod(obj.Oshape)).reshape(1, *obj.Oshape)
    >>> grad_x = obj.backward(grad, x, y)
    >>> print(grad_x)
    [[[[0.         0.         0.         0.        ]
       [0.         0.         0.         0.33333333]
       [0.         0.         0.         0.        ]
       [0.         0.66666667 0.         1.        ]]]]
    """
    Ishape: Tuple = field(init=False, default_factory=tuple)
    Oshape: Tuple = field(init=False, default_factory=tuple)
    stride: Union[int, Tuple] = field(init=False, default_factory=tuple)
    pad: Union[int, Tuple] = 0

    def __post_init__(self, *args, **kwds):
        self.Ishape = _get_shape(self.prev)
        self.stride = _get_stride(self.n)
        self.pad = _get_pad(self.pad)
        self.Oshape = get_Oshape(self.Ishape, self.stride,
                                 self.stride, self.pad)
        self.prev = np.prod(self.Ishape)
        self.n = np.prod(self.Oshape)

    def forward(self, x, *args, **kwds):
        """Forward ropagation.

        Args:
            x (ndarray): Input array.
                         4D-array -> 2D-array -> 2D-array
                         (B, C, Ih, Iw) -> (CFhFw, BOhOw) -> (BOhOwC, FhFw)

        Returns:
            y (ndarray): Output array after activation.
                         2D-array -> 4D-array -> 4D-array
                         (BOhOwM, 1) -> (B, Oh, Ow, M) -> (B, M, Oh, Ow)
        """
        B = x.shape[0]
        M, Oh, Ow = self.Oshape

        x = im2col(x, self.stride, stride=self.stride, pad=self.pad)
        x = x.T.reshape(B*Oh*Ow*M, -1)
        y = np.max(x, axis=1, keepdims=True)
        self.max_index = np.where(y==x, 1, 0)
        y = y.reshape(B, Oh, Ow, M).transpose(0, 3, 1, 2)
        return y

    def backward(self, grad, x, y, *args, **kwds):
        """Backward propagation.

        Args:
            grad (ndarray): Gradient flowing from the lower layer.
                            4D-array -> 4D-array -> 2D-array
                            (B, M, Oh, Ow) -> (B, Oh, Ow, M) -> (BOhOwM, 1)
            x (ndarray): Input array.
                         4D-array -> (B, C, Ih, Iw)
            y (ndarray): Output array after activation.
                         4D-array -> 4D-array -> 2D-array
                         (B, M, Oh, Ow) -> (B, Oh, Ow, M) -> (BOhOw, M)

        Returns:
            grad_x (ndarray): Gradient flowing to the upper layer.
                              2D-array -> 2D-array -> 4D-array
                              (BOhOwC, FhFw) -> (CFhFw, BOhOw)
                                             -> (B, C, Ih, Iw)
        """
        B = grad.shape[0]
        M, Oh, Ow = self.Oshape

        grad = grad.transpose(0, 2, 3, 1).reshape(-1, 1)
        grad_x = self.max_index*grad
        grad_x = grad_x.reshape(B*Oh*Ow, M*np.prod(self.stride)).T
        grad_x = col2im(grad_x, x, self.stride,
                        stride=self.stride, pad=self.pad)
        return grad_x

    def update(self, *args, **kwds):
        """Update function.
        No processing.
        """
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
