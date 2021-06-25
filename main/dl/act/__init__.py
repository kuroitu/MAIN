from ._interface import get_act
from ._base import BaseAct
from ._step import Step
from ._identity import Identity
from ._bent_identity import BentIdentity
from ._hard_shrink import HardShrink
from ._soft_shrink import SoftShrink
from ._threshold import Threshold
from ._sigmoid import Sigmoid
from ._hard_sigmoid import HardSigmoid
from ._log_sigmoid import LogSigmoid
from ._tanh import Tanh
from ._tanh_shrink import TanhShrink
from ._hard_tanh import HardTanh
from ._relu import ReLU
from ._relu6 import ReLU6
from ._lrelu import LReLU
from ._elu import ELU
from ._selu import SELU
from ._celu import CELU
from ._softmax import Softmax
from ._softmin import Softmin
from ._log_softmax import LogSoftmax
from ._softplus import Softplus
from ._softsign import Softsign
from ._swish import Swish
from ._mish import Mish
from ._tanh_exp import TanhExp
from ._acon import ACON


act_dict = {"step":Step,
            "identity": Identity,
            "bentidentity": BentIdentity,
            "hardshrink": HardShrink,
            "softshrink": SoftShrink,
            "threshold": Threshold,
            "sigmoid": Sigmoid,
            "hardsigmoid": HardSigmoid,
            "logsigmoid": LogSigmoid,
            "tanh": Tanh,
            "tanhshrink": TanhShrink,
            "hardtanh": HardTanh,
            "relu": ReLU,
            "relu6": ReLU6,
            "leakyrelu": LReLU, "lrelu": LReLU,
            "elu": ELU,
            "selu": SELU,
            "celu": CELU,
            "softmax": Softmax,
            "softmin": Softmin,
            "logsoftmax": LogSoftmax,
            "softplus": Softplus,
            "softsign": Softsign,
            "swish": Swish,
            "mish": Mish,
            "tanhexp": TanhExp,
            "acona": ACON, "aconb": ACON, "aconc": ACON}

