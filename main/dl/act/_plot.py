from typing import Any, Dict

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from mypy_extensions import NoReturn

try:
    from ._interface import get_act
except ImportError:
    # For doctest
    from main.dl.act import get_act


def plot_graph(x: ndarray, name: str, *args: Any,
               save: bool = True, place: str = ".", **kwds: Any) -> NoReturn:
    """
    Normal plot for activation functions.

    Args:
        x (ndarray): Input values.
        name (str): Name of activation function.
        save (bool): If save graph or not.
        place (str): Save directory.
    """
    obj = get_act(name, *args, **kwds)
    y = obj.forward(x, *args, **kwds)
    dy = obj.backward(x, y, *args, **kwds)

    fig, ax = plt.subplots(1)
    ax.plot(x, y, label="forward")
    ax.plot(x, dy, label="backward")
    fig.suptitle(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y, dy")
    ax.grid()
    ax.legend(loc="best")
    if save:
        fig.savefig(f"{place}/{name}.png")
    fig.show()


def param_plot(x: ndarray, name: str, param_dict: Dict, *args: Any,
               save: bool = True, place: str = ".", **kwds: Any) -> NoReturn:
    """
    Plot graph for difference parameters.

    Args:
        x (ndarray): Input values.
        name (str): Name of activation function.
        param_dict (dict): Parameter dictionary.
        save (bool): If save graph or not.
        place (str): Save directory.
    """
    for key in param_dict:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for value in param_dict[key]:
            kwds[key] = value
            obj = get_act(name, *args, **kwds)
            y = obj.forward(x, *args, **kwds)
            dy = obj.backward(x, y, *args, **kwds)

            ax[0].plot(x, y, label=f"{key}={np.round(value, 2)}")
            ax[1].plot(x, dy, label=f"{key}={np.round(value, 2)}")
        start = param_dict[key][0]
        end = param_dict[key][-1]
        step = param_dict[key][1]-param_dict[key][0]
        fig.suptitle(f"{name} {key}={np.round(start, 2)}~"
                     f"{np.round(end, 2)}"
                     f"step by {np.round(step, 2)}")
        ax[0].set_title("forward")
        ax[1].set_title("backward")
        ax[0].set_xlabel("x")
        ax[1].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[1].set_ylabel("dy")
        ax[0].grid()
        ax[1].grid()
        ax[0].legend(loc="best")
        ax[1].legend(loc="best")

        if save:
            fig_name = "_".join([name, key, str(np.round(start, 2)),
                                            str(np.round(end, 2)),
                                            str(np.round(step, 2))])
            fig.savefig(f"{place}/{fig_name}.png")
