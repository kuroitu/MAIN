import sys
from typing import Union, Any, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    from ._base import BaseBlock
except ImportError:
    # For doctest
    from main.dl.layer._base import BaseBlock


@dataclass
class Output(BaseBlock):
    """Output layer class.
    In forward propagation, error calculation is performed in particular.
    In backward propagation, error backpropagation and
    activation backpropagation may be skipped, so the branch is implemented
    in particular.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> prev, n = 3, 2
    >>> x = np.linspace(0, 1, prev).reshape(1, prev)
    >>> t = np.linspace(0, 1, n).reshape(1, n)

    >>> obj = Output(prev=prev, n=n, act="sigmoid", err="binary")
    >>> print(obj)
    Output(prev=3, n=2, act=Sigmoid(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Binary(eps=1e-08), err_args=(), err_kwds={})
    >>> u, y = obj.forward(x, t=t)
    >>> print(u, y)
    [[ 0.16535077 -0.00040942]] [[0.54124377 0.49989764]]
    >>> grad_x = obj.backward(t, x, u, y)
    >>> print(grad_x)
    [[-0.04316482 -0.07445177 -0.01065315]]

    >>> obj = Output(prev=prev, n=n, act="softmax", err="cross")
    >>> print(obj)
    Output(prev=3, n=2, act=Softmax(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Cross(eps=1e-08), err_args=(), err_kwds={})
    >>> u, y = obj.forward(x, t=t)
    >>> print(u, y)
    [[0.06384614 0.0591243 ]] [[0.50118046 0.49881954]]
    >>> grad_x = obj.backward(t, x, u, y)
    >>> print(grad_x)
    [[-0.00676985 -0.03512162 -0.01931639]]

    >>> obj = Output(prev=prev, n=n, act="identity", err="square")
    >>> print(obj)
    Output(prev=3, n=2, act=Identity(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Square(), err_args=(), err_kwds={})
    >>> u, y = obj.forward(x, t=t)
    >>> print(u, y)
    [[-0.07660099 -0.02577971]] [[-0.07660099 -0.02577971]]
    >>> grad_x = obj.backward(t, x, u, y)
    >>> print(grad_x)
    [[ 0.00479996  0.04260664 -0.02374536]]
    """
    err: Union[str, Any] = "square"
    err_args: Tuple = field(default_factory=tuple)
    err_kwds: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, *args, **kwds):
        from main.dl.err import get_err, Cross, Binary
        from main.dl.act import Softmax, Sigmoid

        super().__post_init__(*args, **kwds)

        self.err = get_err(self.err, *self.err_args, **self.err_kwds)

        if (isinstance(self.act, Softmax)
            and isinstance(self.err, Cross)) \
        or (isinstance(self.act, Sigmoid)
            and isinstance(self.err, Binary)):
            # If (Softmax, Cross Entropy) pair
            # or (Sigmoid, Binary Cross Entroy) pair are had this layer,
            # then skip activation backpropagation and error backpropagation.
            self.backward = self._skip_backward

    def forward(self, x, *args, t=None, **kwds):
        """Forward propagation.

        Args:
            x (ndarray): Input array.
            t (ndarray): Theacher signal for one-hot label representation.

        Returns:
            u (ndarray): Output array before activation.
            y (ndarray): Output array after activation.
        """
        u, y = super().forward(x, *args, **kwds)
        if t is not None:
            self.error = self.err.forward(y, t, *args, **kwds)
        return u, y

    def backward(self, t, x, u, y, *args, **kwds):
        """Backward propagation.

        Args:
            t (ndarray): Theacher signal for one-hot label representation.
            x (ndarray): Input array.
            u (ndarray): Inferred value before activation.
            y (ndarray): Inferred value after activation.

        Returns:
            _ (ndarray): Gradient flowing to the upper layer.
        """
        grad = self.err.backward(y, t)
        return super().backward(grad, x, u, y, *args, **kwds)

    def _skip_backward(self, x, u, y, t, *args, **kwds):
        dact = y - t
        grad_x, self.grad = self.fc.backward(x, self.w, dact)
        return grad_x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
