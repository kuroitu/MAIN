import sys
from typing import Union, Tuple, Dict, Any
from dataclasses import dataclass, InitVar, field

import numpy as np


@dataclass
class BaseLayer():
    """Base layer class.
    Implement the common content of all layers.
    """
    prev: Union[int, Tuple] = -1
    n: Union[int, Tuple] = -1
    name: InitVar[str] = ""

    def __post_init__(self, name, *args, **kwds):
        if self.prev <= 0:
            raise ValueError(f"Invalid number of before neurons; {self.prev=}")
        if self.n <= 0:
            raise ValueError(f"Invalid number of neurons; {self.n=}")
        if len(name):
            self.name = name
        else:
            self.name = str(type(self))

    def forward(self, x, *args, **kwds):
        raise NotImplementedError("'forward' method must be implemented.")

    def backward(self, grad, x, u, y, *args, **kwds):
        raise NotImplementedError("'backward' method must be implemented.")

    def update(self, *args, **kwds):
        raise NotImplementedError("'update' method must be implemented.")


@dataclass
class BaseBlock(BaseLayer):
    """Base block class.
    Implement the processing common to almost all layers.

    Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> prev, n = 3, 2
    >>> obj = BaseBlock(prev=prev, n=n, act="sigmoid")
    >>> print(obj)
    BaseBlock(prev=3, n=2, act=Sigmoid(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})
    >>> x = np.linspace(0, 1, prev).reshape(1, prev)
    >>> u, y = obj.forward(x)
    >>> print(u, y)
    [[ 0.16535077 -0.00040942]] [[0.54124377 0.49989764]]
    >>> grad = np.linspace(0, 1, n).reshape(1, n)
    >>> grad_x = obj.backward(grad, x, u, y)
    >>> print(grad_x)
    [[ 0.00500196  0.02801116 -0.01221597]]
    """
    wb_width: InitVar[float] = 5e-2
    act: Union[str, Any] = "relu"
    opt: Union[str, Any] = "adam"
    act_args: Tuple = field(default_factory=tuple)
    opt_args: Tuple = field(default_factory=tuple)
    act_kwds: Dict[str, Any] = field(default_factory=dict)
    opt_kwds: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, name, wb_width, *args, **kwds):
        """
        Attributes:
            params (ndarray): Parameters to be adjusted by gradient descent.
                                w: (prev, n)
                                b: (n,)
                              So this has the shape (prev+1, n).
                              Accessed like this;
                                w: params[:prev] -> (prev, n)
                                b: params[prev] -> (1, n)
        """
        from main.dl.act import get_act
        from main.dl.opt import get_opt
        try:
            from ._fc import FullyConnected
        except ImportError:
            # For doctest
            from main.dl.layer._fc import FullyConnected

        self.params = wb_width*np.random.randn(self.prev+1, self.n)
        self.fc = FullyConnected()
        self.act = get_act(self.act, *self.act_args, **self.act_kwds)
        self.opt = get_opt(self.opt, *self.opt_args, **self.opt_kwds)

        super().__post_init__(name, *args, **kwds)

    def forward(self, x, *args, **kwds):
        """Forward propagation.

        Args:
            x (ndarray): Input array.

        Returns:
            u (ndarray): Output array before activation.
            y (ndarray): Output array after activation.
        """
        u = self.fc.forward(x, self.w, self.b)
        y = self.act.forward(u, *args, **kwds)
        return u, y

    def backward(self, grad, x, u, y, *args, **kwds):
        """Backward propagation.

        Args:
            grad (ndarray): Gradient flowing from the lower layer.
            x (ndarray): Input array.
            u (ndarray): Output array before activation.
            y (ndarray): Output array after activation.

        Returns:
            grad_x (ndarray): Gradient flowing to the upper layer.
        """
        dact = grad*self.act.backward(u, y, *args, **kwds)
        grad_x, self.grad = self.fc.backward(x, self.w, dact)
        return grad_x

    def update(self, *args, **kwds):
        """Update parameters."""
        delta = self.opt.update(self.grad, *args, **kwds)
        self.params += self.delta

    @property
    def w(self):
        return self.params[:self.prev]

    @w.setter
    def w(self, value):
        self.params[:self.prev] = value

    @property
    def b(self):
        return self.params[self.prev]

    @b.setter
    def b(self, value):
        self.params[self.prev] = value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
