import numpy as np

try:
    from ._base import BaseLayer
except ImportError:
    # For doctest
    from main.dl.layer._base import BaseLayer


class FullyConnected(BaseLayer):
    def forward(self, x, w, b, *args, **kwds):
        return x@w + b

    def backward(self, x, w, grad, *args, **kwds):
        grad_x = grad@w.T
        grad_w = x.T@grad
        grad_b = np.sum(grad, axis=0, keepdims=True)
        grad_params = np.vstack((grad_w, grad_b))
        return grad_x, grad_params

    def update(self, *args, **kwds):
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
