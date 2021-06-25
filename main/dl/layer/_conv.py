import sys
from typing import Union, Tuple
from dataclasses import dataclass, field

import numpy as np

from main.dl.util import im2col, col2im, get_Oshape, get_same_pad
from main.dl.util._util import _get_shape, _get_stride, _get_pad
try:
    from ._base import BaseBlock
except:
    # For doctest
    from main.dl.layer._base import BaseBlock


@dataclass
class Convolution(BaseBlock):
    """Convolution layer class.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> prev, n = (4, 4), (2, 2) # 'prev' means input array's shape, 'n' means filter array's shape.
    >>> obj = Convolution(prev=prev, n=n, act="sigmoid")
    >>> print(obj)
    Convolution(prev=4, n=1, act=Sigmoid(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, Ishape=(1, 1, 4, 4), Fshape=(1, 1, 2, 2), Oshape=(1, 5, 5), stride=(1, 1), pad=(1, 1))
    >>> x = np.linspace(0, 1, np.prod(prev)).reshape(1, 1, *prev)
    >>> x, u, y = obj.forward(x)
    >>> print(x)
    [[0.         0.         0.         0.         0.         0.
      0.         0.06666667 0.13333333 0.2        0.         0.26666667
      0.33333333 0.4        0.46666667 0.         0.53333333 0.6
      0.66666667 0.73333333 0.         0.8        0.86666667 0.93333333
      1.        ]
     [0.         0.         0.         0.         0.         0.
      0.06666667 0.13333333 0.2        0.         0.26666667 0.33333333
      0.4        0.46666667 0.         0.53333333 0.6        0.66666667
      0.73333333 0.         0.8        0.86666667 0.93333333 1.
      0.        ]
     [0.         0.         0.06666667 0.13333333 0.2        0.
      0.26666667 0.33333333 0.4        0.46666667 0.         0.53333333
      0.6        0.66666667 0.73333333 0.         0.8        0.86666667
      0.93333333 1.         0.         0.         0.         0.
      0.        ]
     [0.         0.06666667 0.13333333 0.2        0.         0.26666667
      0.33333333 0.4        0.46666667 0.         0.53333333 0.6
      0.66666667 0.73333333 0.         0.8        0.86666667 0.93333333
      1.         0.         0.         0.         0.         0.
      0.        ]]
    >>> print(u)
    [[[[0.0933779  0.10084754 0.11157965 0.12231175 0.10316528]
       [0.12325648 0.14510982 0.16305595 0.18100209 0.13385564]
       [0.15847048 0.21689436 0.2348405  0.25278663 0.17042618]
       [0.19368449 0.2886789  0.30662504 0.32457117 0.20699672]
       [0.10938419 0.18128014 0.18849417 0.1957082  0.18158052]]]]
    >>> print(y)
    [[[[0.52332753 0.52519054 0.52786601 0.53053987 0.52576847]
       [0.53077517 0.53621393 0.54067391 0.54512738 0.53341403]
       [0.53953492 0.55401201 0.55844178 0.56286227 0.54250372]
       [0.54827032 0.57167268 0.57606126 0.58043788 0.55156519]
       [0.52731881 0.54519633 0.54698451 0.54877148 0.54527081]]]]
    >>> grad = np.linspace(0, 1, np.prod(obj.Oshape)).reshape(1, *obj.Oshape)
    >>> grad_x = obj.backward(grad, x, u, y)
    >>> print(grad_x)
    [[[[0.03833207 0.03830644 0.01570653 0.01817381]
       [0.04267863 0.0180307  0.01974008 0.04558564]
       [0.02023373 0.02109517 0.04875266 0.05068746]
       [0.02240818 0.05182163 0.05469077 0.02456746]]]]
    """
    Ishape: Tuple = field(init=False, default_factory=tuple)
    Fshape: Tuple = field(init=False, default_factory=tuple)
    Oshape: Tuple = field(init=False, default_factory=tuple)
    stride: int = 1
    pad: Union[int, str, Tuple] = "same"

    def __post_init__(self, *args, **kwds):
        self.Ishape = _get_shape(self.prev)
        self.Fshape = _get_shape(self.n)
        self.prev = np.prod(self.Fshape[1:])
        self.n = self.Fshape[0]
        self.stride = _get_stride(self.stride)
        if self.pad == "same":
            self.pad = get_same_pad(self.Ishape, self.Fshape, self.stride)
        else:
            self.pad = _get_pad(self.pad)
        self.Oshape = get_Oshape(self.Ishape, self.Fshape,
                                 self.stride, self.pad)

        super().__post_init__(*args, **kwds)

    def forward(self, x, *args, **kwds):
        """Forward ropagation.

        Args:
            x (ndarray): Input array.
                         4D-array -> (B, C, Ih, Iw)

        Returns:
            u (ndarray): Output array before activation.
                         2D-array -> 4D-array -> 4D-array
                         (BOhOw, M) -> (B, Oh, Ow, M) -> (B, M, Oh, Ow)
            y (ndarray): Output array after activation.
                         2D-array -> 4D-array -> 4D-array
                         (BOhOw, M) -> (B, Oh, Ow, M) -> (B, M, Oh, Ow)
        """
        B = x.shape[0]
        M, Oh, Ow = self.Oshape

        x = im2col(x, self.Fshape, stride=self.stride, pad=self.pad)
        u, y = super().forward(x.T, *args, **kwds)
        u = u.reshape(B, Oh, Ow, M).transpose(0, 3, 1, 2)
        y = y.reshape(B, Oh, Ow, M).transpose(0, 3, 1, 2)
        return x, u, y

    def backward(self, grad, x, u, y, *args, **kwds):
        """Backward propagation.

        Args:
            grad (ndarray): Gradient flowing from the lower layer.
                            4D-array -> 4D-array -> 2D-array
                            (B, M, Oh, Ow) -> (B, Oh, Ow, M) -> (BOhOw, M)
            x (ndarray): Input array.
                         4D-array -> (B, C, Ih, Iw)
            u (ndarray): Output array before activation.
                         4D-array -> 4D-array -> 2D-array
                         (B, M, Oh, Ow) -> (B, Oh, Ow, M) -> (BOhOw, M)
            y (ndarray): Output array after activation.
                         4D-array -> 4D-array -> 2D-array
                         (B, M, Oh, Ow) -> (B, Oh, Ow, M) -> (BOhOw, M)

        Returns:
            grad_x (ndarray): Gradient flowing to the upper layer.
                              2D-array -> 4D-array
                              (CFhFw, BOhOw) -> (B, C, Ih, Iw)
        """
        B = grad.shape[0]
        M, Oh, Ow = self.Oshape

        grad = grad.transpose(0, 2, 3, 1).reshape(B*Oh*Ow, M)
        u = u.transpose(0, 2, 3, 1).reshape(B*Oh*Ow, M)
        y = y.transpose(0, 2, 3, 1).reshape(B*Oh*Ow, M)
        grad_x = super().backward(grad, x.T, u, y, *args, **kwds)
        grad_x = col2im(grad_x, self.Ishape, self.Fshape,
                        stride=self.stride, pad=self.pad)
        return grad_x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
