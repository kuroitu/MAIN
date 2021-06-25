from ._interface import get_err
from ._base import BaseError
from ._square import Square
from ._binary import Binary
from ._cross import Cross


err_dict = {"square": Square, "binary": Binary, "cross": Cross}
