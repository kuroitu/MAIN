import numpy as np
from pyannotate_runtime import collect_types

try:
    from . import err_dict
    from ._base import BaseError
    from ._interface import get_err
except:
    # For doctest
    from main.dl.err import BaseError, err_dict, get_err


def main() -> None:
    y = np.array([0, 0.5, 1])
    t = np.array([0, 1, 0])
    err_dict["base"] = BaseError
    for err in _err_dict:
        obj = get_err(err)
        try:
            obj.forward(y, t)
        except:
            pass
        try:
            obj.backward(y, t)
        except:
            pass


if __name__ == "__main__":
    collect_types.init_types_collection()
    with collect_types.collect():
        main()
    collect_types.dump_stats("_type_info.json")
