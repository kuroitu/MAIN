import sys
from typing import Union, Any, Dict
from dataclasses import dataclass, field

import numpy as np

from main.dl.opt import get_opt, BaseOpt
try:
    from ._base import BaseAct
except ImportError:
    # For doctest
    from main.dl.act import BaseAct


@dataclass
class ACON(BaseAct):
    """
    ACON-Activation function class.
    Let me as follow;
        f(x) = (p1-p2) * x * sigma(beta * (p1-p2) * x) + p2 * x
    """
    acon_type: str = "aconc"
    p1: float = 1
    p2: float = 0.25
    beta: float = 1
    opt: Union[str, BaseOpt] = "adam"
    opt_dict: Dict[str, Any] = field(default_factory=dict)
    switch_dict: Dict[str, Any] = field(default_factory=dict)
    switch_opt: Union[str, BaseOpt] = "adam"
    switch_opt_dict: Dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        """Initialize method after 'dataclass' initialize."""
        if self.acon_type == "acona":
            self.p2 = 0
        elif not self.acon_type in ["aconb", "aconc"]:
            raise KeyError(f"Unknown ACON type: <'{self.acon_type}'>")
        else:
            self.opt = get_opt(self.opt, **self.opt_dict)

        @dataclass
        class _BaseSwitch():
            """Base switch class."""
            opt : Union[str, BaseOpt] = "adam"
            opt_dict : Dict[str, Any] = field(default_factory=dict)
            def forward(self, *args, **kwds):
                raise NotImplementedError(
                        "'forward' method must be implemented.")
            def backward(self, *args, **kwds):
                pass
            def update(self, *args, **kwds):
                pass
        if isinstance(self.beta, str):
            if self.beta == "pixel":
                class _Beta(_BaseSwitch):
                    """Pixel-wise switch class."""
                    def forward(self, x, *args, **kwds):
                        self.y = 1/(1+np.exp(-x))
            elif self.beta == "channel":
                if not "C" in self.switch_dict:
                    raise KeyError(
                            "Must need 'C' value to use channel-wise switch.")
                @dataclass
                class _Beta(BaseSwitch):
                    """Channel-wise switch class."""
                    C : int
                    r : int = 16
                    width : float = 5e-2
                    def __post_init__(self):
                        self.w1 = self.width*np.random.randn(
                                (self.C, self.C//self.r))
                        self.w2 = self.width*np.random.randn(
                                (self.C//self.r, self.C))
                        self.opt = get_opt(self.opt_type, **self.opt_dict)
                    def forward(self, x, *args, **kwds):
                        self.sumx = np.sum(x, axis=(2, 3))
                        self.first = self.sumx@self.w1
                        self.y = 1/(1+np.exp(-self.first@self.w2))
                    def backward(self, grad, *args, **kwds):
                        self.grad_w2 \
                                = self.first.T@(grad*self.y*(1-self.y))
                        self.grad_w1 \
                                = (self.sumx.T
                                   @ (grad*self.y*(1-self.y)@self.w2.T))
                    def update(self, *args, **kwds):
                        dw1, dw2 = self.opt.update(self.grad_w1, self.grad_w2,
                                                   *args, **kwds)
                        self.w1 -= dw1
                        self.w2 -= dw2
            elif self.beta == "layer":
                class _Beta(_BaseSwitch):
                    """Layer-wise switch class."""
                    def forward(self, x, *args, **kwds):
                        self.y = 1/(1+np.exp(-np.sum(x, axis=(1, 2, 3))))
            else:
                raise KeyError(f"Unknown switch type: <{self.beta}>")
            self.beta = _Beta(opt=self.switch_opt_type,
                              opt_dict=self.switch_opt_type,
                              **self.switch_dict)
        elif isinstance(self.beta, int) or isinstance(self.beta, float):
            @dataclass
            class _Beta(_BaseSwitch):
                """Normal value switch class."""
                y : float = 1
            self.beta = _Beta(y=self.beta, **self.switch_dict)
        elif not (hasattr(self.beta, "forward")
              and hasattr(self.beta, "backward")
              and hasattr(self.beta, "update")):
            raise KeyError("Using designed function, the type of switch 'beta'"
                           " must be a <class> and have method 'forward', "
                           "'backward' and 'update'.")

    def forward(self, x, *args, **kwds):
        """Forward propagation.

        Args:
            x (ndarray): Input array before activation.

        Returns:
            _ (ndarray): Activated by 'ACON' function.
        """
        self.beta.forward(x, *args, **kwds)
        self.sig = 1/(1+np.exp(-self.beta.y*(self.p1-self.p2)*x))
        return (self.p1-self.p2)*x*self.sig + self.p2*x


    def backward(self, x, grad, *args, **kwds):
        """Backward propagation.

        Args:
            x (ndarray): Input array before activation.
            grad (ndarray): Gradient values flowing from the upper layer.

        Returns:
            _ (ndarray): Partial derivative with respect to x.
        """
        if self.acon_type in ["aconb", "aconc"]:
            self.grad_p1 = np.sum(grad*self.sig*(2-self.sig)*self.beta.y*x)
            self.grad_p2 = np.sum(grad*x) - self.grad_p1
        self.beta.backward(grad)
        return (((self.p1-self.p2)*(1+np.exp(-self.beta.y*(self.p1-self.p2)*x))
                 + self.beta.y * (self.p1-self.p2)**2
                 * np.exp(-self.beta.y*(self.p1-self.p2)*x)*x)
                /(1+np.exp(-self.beta.y*(self.p1-self.p2)*x))**2
                +self.p2)


    def update(self, *args, **kwds):
        """Update parameters."""
        if self.acon_type in ["aconb", "aconc"]:
            dp1, dp2 = self.opt.update(self.grad_p1, self.grad_p2,
                                       *args, **kwds)
            if self.acon_type == "aconc":
                self.p1 -= dp1
            self.p2 -= dp2
        self.beta.update(*args, **kwds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
