import re
import itertools
from typing import Dict, List, Any
from dataclasses import dataclass, field

import numpy as np

from main.dl.layer import layer_dict, BaseLayer


@dataclass
class LayerManager():
    """For managing layers.

    Examples:
    >>> from pprint import pprint
    >>> from main.dl.layer import layer_dict
    >>> lm = LayerManager()
    >>> lm["middle"] = {}
    >>> lm["middle"] = layer_dict["middle"]()
    >>> lm["output"] = {}
    >>> pprint(lm.names)
    {'convolution': [],
     'dropout': [],
     'middle': ['middle1', 'middle2'],
     'output': ['output1'],
     'pooling': []}

    >>> lm["middle1"] = {"act": "sigmoid"}
    >>> pprint(lm.layers)
    {'convolution': [],
     'dropout': [],
     'middle': [Middle(prev=1, n=1, act=Sigmoid(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}),
                Middle(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})],
     'output': [Output(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Square(), err_args=(), err_kwds={})],
     'pooling': []}

    >>> pprint(lm["middle1"])
    Middle(prev=1, n=1, act=Sigmoid(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})
    >>> pprint(lm[1])
    Middle(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})

    >>> print(len(lm))
    3
    >>> print(lm.each_len())
    {'middle': 2, 'dropout': 0, 'output': 1, 'convolution': 0, 'pooling': 0}

    >>> print(lm.index("middle2"))
    [1]

    >>> del lm[0]
    >>> pprint(lm.names)
    {'convolution': [],
     'dropout': [],
     'middle': ['middle1'],
     'output': ['output1'],
     'pooling': []}
    >>> pprint(lm.layers)
    {'convolution': [],
     'dropout': [],
     'middle': [Middle(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})],
     'output': [Output(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Square(), err_args=(), err_kwds={})],
     'pooling': []}

    >>> print("middle1" in lm)
    True
    >>> print("middle2" in lm)
    False
    >>> for l in lm:
    ...     print(l)
    middle1
    output1
    >>> for l in reversed(lm):
    ...     print(l)
    output1
    middle1
    >>> for key in lm.keys():
    ...     print(key)
    middle1
    output1
    >>> for value in lm.values():
    ...     print(value)
    Middle(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})
    Output(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Square(), err_args=(), err_kwds={})
    >>> for key, value in lm.items():
    ...     print(key)
    ...     print(value)
    middle1
    Middle(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})
    output1
    Output(prev=1, n=1, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}, err=Square(), err_args=(), err_kwds={})

    >>> lm2 = LayerManager()
    >>> lm2.append(name="middle", prev=1, n=10)
    >>> lm2.append(value=layer_dict["middle"](name="middle", prev=10, n=10, act="mish"))
    >>> lm2.append()
    Traceback (most recent call last):
        ...
    KeyError: "Unspecified layer's type."
    >>> lm2.insert(0, name="middle", prev=10, n=10, act="tanhexp")
    >>> pprint(lm2.layers)
    {'convolution': [],
     'dropout': [],
     'middle': [Middle(prev=1, n=10, act=ReLU(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}),
                Middle(prev=10, n=10, act=Mish(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={}),
                Middle(prev=10, n=10, act=TanhExp(), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), act_args=(), opt_args=(), act_kwds={}, opt_kwds={})],
     'output': [],
     'pooling': []}
    >>> lm.extend(lm2)
    >>> pprint(lm.names)
    {'convolution': [],
     'dropout': [],
     'middle': ['middle1', 'middle2', 'middle3', 'middle4'],
     'output': ['output1'],
     'pooling': []}
    >>> lm.extend_insert(0, lm2)
    >>> pprint(lm.names)
    """
    __calc_order: List = field(init=False, default_factory=list)
    __names: Dict[str, List] = field(init=False, default_factory=dict)
    __layers: Dict[str, List] = field(init=False, default_factory=dict)

    def __post_init__(self, *args, **kwds):
        self.__calc_order = []
        self.__names = {ktype: [] for ktype in layer_dict}
        self.__layers = {ktype: [] for ktype in layer_dict}

    def __len__(self):
        return np.sum([len(layer) for layer in self.__layers.values()])

    def __getitem__(self, key):
        if isinstance(key, str):
            ktype = self._extract_type(key)
            idx = self.__names[ktype].index(key)
        elif isinstance(key, int):
            ktype = self._extract_type(self.__calc_order[key])
            idx = key
        else:
            raise KeyError(f"Unknown key type; {key} = {type(key)}")

        return self.__layers[ktype][idx]

    def __setitem__(self, key, value):
        if isinstance(key, str) or isinstance(key, int):
            key = self._create_key(key)
        else:
            raise KeyError(f"Unknown key type; {key} = {type(key)}")
        value = self._create_value(key, value)

        if key not in self.__calc_order:
            self._add(key, value)
        else:
            self._overwrite(key, value)

    def __delitem__(self, key):
        if isinstance(key, str):
            ktype = self._extract_type(key)
            idx = self._extract_num(key) - 1
            indices = self.index(key)
            for count, i in enumerate(indices):
                del self.__calc_order[i-count]
            del self.__names[ktype][idx]
            del self.__layers[ktype][idx]
            for l in range(idx, len(self.__names[ktype])):
                this_idx = self._extract_num(self.__names[ktype][l]) - 1
                self.__names[ktype][l] = f"{ktype}{this_idx}"
                self.__layers[ktype][l].name = f"{ktype}{tihs_idx}"
        elif isinstance(key, int):
            ktype = self._extract_type(self.__calc_order[key])
            idx = self._extract_num(self.__calc_order[key])-1
            indices = self.index(self.__calc_order[key])
            for count, i in enumerate(indices):
                del self.__calc_order[i-count]
            del self.__names[ktype][idx]
            del self.__layers[ktype][idx]
        elif isinstance(key, slice):
            for count, i in enumerate(range(key.start, key.end, key.step)):
                del self[i-count]
        elif isinstance(key, list) or isinstance(key, tuple):
            for count, i in enumerate(key):
                del self[i-count]
        else:
            raise keyError(f"Unknown key type; {key} = {type(key)}")

        self._rename()

    def __contains__(self, key):
        return key in set(self.__calc_order)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self):
            raise StopIteration
        self._i += 1
        return self.__calc_order[self._i-1]

    def __reversed__(self):
        self._i = 0
        return self.__calc_order[::-1]

    def __add__(self, other):
        if isinstance(other, LayerManager):
            return self.extend(other)
        elif isinstance(other, BaseLayer):
            return self.append(value=other)

    @property
    def calc_order(self):
        return self.__calc_order

    @property
    def names(self):
        return self.__names

    @property
    def layers(self):
        return self.__layers

    def index(self, key):
        return [idx for idx, k in enumerate(self.__calc_order) if k == key]

    def each_len(self):
        return {ktype: len(layer) for ktype, layer in self.__layers.items()}

    def keys(self):
        keys = []
        for ktype in self.__names:
            keys.extend(self.__names[ktype])
        return keys

    def values(self):
        values = []
        for ktype in self.__layers:
            values.extend(self.__layers[ktype])
        return values

    def items(self):
        return zip(self.keys(), self.values())

    def append(self, *args, value=None, **kwds):
        if value is None:
            if "name" not in kwds:
                raise KeyError("Unspecified layer's type.")
            if len(self) == 0 and not "prev" in kwds:
                raise KeyError("The first layer must be specified "
                               "'prev' value.")
            if len(self) != 0 and "prev" in kwds:
                key = self.__calc_order[-1]
                ktype = self._extract_type(key)
                idx = self._extract_num(key) - 1
                if self.__layers[ktype][idx].n != kwds["prev"]:
                    raise ValueError(f"Unmatch 'prev' value; "
                                     f"expected = {self.__layers[ktype][idx].n} "
                                     f"prev = {kwds['prev']}")
        else:
            if not isinstance(value, BaseLayer):
                raise TypeError(f"Unexpected 'value' type; "
                                f"expected 'BaseLayer' but "
                                f"'{type(value)} has come.")
            value.name = self._extract_type(value.name)

        if value is None:
            key = self._create_key(kwds["name"])
            value = self._create_value(key, kwds)
        else:
            key = self._create_key(value.name)
            value.name = key
        self._add(key, value)

    def extend(self, other):
        if not isinstance(other, LayerManager):
            raise TypeError(f"Unexpected object type; {type(other)=}")
        if len(other) == 0:
            raise ValueErorr(f"There is no layer to add.")

        current_order = self.__calc_order
        corres_table = {}
        for key, value in other.items():
            # Initialize key and value's name, then add.
            self.append(value=value)
            corres_table[key] = self.__calc_order.pop()
        for key in other.calc_order:
            # Add while maintaining the calculation order.
            self.__calc_order.append(corres_table[key])

    def insert(self, upper, *args, value=None, **kwds):
        if value is None:
            if "name" not in kwds:
                raise KeyError("Unspecified layer's type.")
            if isinstance(upper, BaseLayer):
                upper = upper.name
        else:
            if not isinstance(value, BaseLayer):
                raise TypeError(f"Unexpected 'value' type; "
                                f"expected 'BaseLayer' but "
                                f"{type(value)} has come.")
        if idx := self._insert_place(upper) < 0:
            raise IndexError("Fail to insert value.")

        if value is not None:
            self.append(value=value)
        else:
            self.append(**kwds)
        key = self.__calc_order.pop()
        self.__calc_order.insert(idx, key)

    def extend_insert(self, upper, other, *args, **kwds):
        if not isinstance(other, LayerManager):
            raise TypeError(f"Unexpected object type; {type(other)=}")
        if len(other) == 0:
            raise ValueErorr(f"There is no layer to add.")
        if idx := self._insert_place(upper) < 0:
            raise TypeError(f"Fail to insert value.")

        current_order = self.__calc_order
        corres_table = {}
        for key, value in other.items():
            ktype = self._extract_type(key)
            value.name = str(type(value))
            self._add(ktype, value)
            corres_table[key] = self.__calc_order.pop()
        for key in other.calc_order:
            self.__calc_order.insert(idx, corres_table[key])
            idx += 1

    def _insert_place(self, upper):
        if isinstance(upper, str):
            if upper not in self.__calc_order:
                raise ValueError(f"The specified layer dose not exist; "
                                 f"{upper=}")

            indices = self.index(upper)
            if len(indices) == 1:
                idx = indices[0]
            elif len(indices) > 1:
                for idx in indices:
                    ans = input(f"Insert after "
                                f"{idx=}'{self.__calc_order[idx]}'? [y/n]")
                    if ans.lower() == "y":
                        break
                else:
                    idx = -1
        elif isinstance(upper, int):
            idx = upper
            if upper < 0:
                idx += len(self)
        else:
            raise TypeError(f"Unknown 'upper' type; {type(upper)=}")

        return idx

    def _extract_type(self, key):
        """
        Examples:
        >>> print(LayerManager()._extract_type("middle1"))
        middle
        """
        return re.sub(r"\d+", "", key)

    def _extract_num(self, key):
        """
        Examples:
        >>> print(LayerManager()._extract_num("middle1"))
        1
        """
        return int(re.sub(r"\D+", "", key))

    def _create_key(self, key):
        if isinstance(key, str):
            if key in self.__calc_order:
                return key
            elif key in layer_dict:
                return f"{key}{len(self.__names[key])+1}"
            else:
                raise KeyError(f"Invalid key; type={type(key)}, {key=}")
        elif isinstance(key, int):
            return self.__calc_order[key]
        else:
            raise KeyError(f"Invalid key; type={type(key)}, {key=}")

    def _create_value(self, key, value):
        ktype = self._extract_type(key)
        if isinstance(value, dict):
            if "name" in value:
                if value["name"] not in [ktype, key]:
                    raise ValueError(f"Different name; "
                                     f"{key=}, {value['name']=}")
            else:
                value["name"] = key
            value = layer_dict[ktype](**value)
        elif isinstance(value, layer_dict[ktype]):
            if value.name == str(type(value)):
                value.name = key
            if key != value.name:
                raise ValueError(f"Different name; "
                                 f"{key=}, {value.name=}")
        else:
            ValueError(f"Invalid value; type={type(value)}, {value=}")
        return value

    def _add(self, key, value):
        ktype = self._extract_type(key)
        self.__calc_order.append(key)
        self.__names[ktype].append(key)
        self.__layers[ktype].append(value)

    def _overwrite(self, key, value):
        ktype = self._extract_type(key)
        idx = self._extract_num(key)-1
        if isinstance(value, layer_dict[ktype]):
            self.__layers[ktype][idx] = value
        else:
            raise TypeError(f"Type of 'key' and 'value' are different; "
                            f"{key=}, {value=}")

    def _rename(self):
        for ktype in layer_dict:
            for i, name in enumerate(self.__names[ktype]):
                idx = self._extract_num(name) - 1
                if idx != i:
                    indices = self.index(name)
                    for j in indices:
                        self.__calc_order[j] = f"{ktype}{idx}"
                    self.__names[ktype][i] = f"{ktype}{idx}"
                    self.__layers[ktype][i].name = f"{ktype}{idx}"


if __name__ == "__main__":
    import doctest
    doctest.testmod()
