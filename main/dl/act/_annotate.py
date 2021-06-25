import numpy as np
from pyannotate_runtime import collect_types

try:
    from . import act_dict
    from ._base import BaseAct
    from ._plot import plot_graph, param_plot
    from ._interface import get_act
except ImportError:
    # For doctest
    from main.dl.act import BaseAct, act_dict, get_act
    from main.dl.act._plot import plot_graph, param_plot


def main() -> None:
    x = np.arange(-10, 10, 5e-2)
    act_dict["base"] = BaseAct
    for act in act_dict:
        obj = get_act(act)
        try:
            y = obj.forward(x)
        except:
            pass
        try:
            dy = obj.backward(x, y)
        except:
            pass
        try:
            obj.update()
        except:
            pass
    plot_graph(x, "step", save=False)
    param_plot(x, "lrelu", {"alpha": np.arange(-1, 0, 5e-2)}, save=False)


if __name__ == "__main__":
    collect_types.init_types_collection()
    with collect_types.collect():
        main()
    collect_types.dump_stats("_type_info.json")
