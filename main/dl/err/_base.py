from typing import Any
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from mypy_extensions import NoReturn


@dataclass
class BaseError():
    """Base class for error functions."""
    def forward(self, *args: ndarray, **kwds: Any) -> NoReturn:
        raise NotImplementedError("'forward' method must be implemented.")

    def backward(self, *args: ndarray, **kwds: Any) -> NoReturn:
        raise NotImplementedError("'backward' method must be implemented.")
