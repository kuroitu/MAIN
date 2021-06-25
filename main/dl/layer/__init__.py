from ._base import BaseLayer
from ._middle import Middle
from ._dropout import Dropout
from ._output import Output
from ._conv import Convolution
from ._pool import Pooling


layer_dict = {"middle": Middle,
              "dropout": Dropout,
              "output": Output,
              "convolution": Convolution,
              "pooling": Pooling}
