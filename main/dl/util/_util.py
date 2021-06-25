import numpy as np
from numpy import ndarray


def _get_shape(array):
    """
    Get shape of array.

    Args:
        array (ndarray/tuple): Array or array's shape.

    Returns:
        shape (tuple): Shape of input array.
                       4D-array: (N, C, H, W)
                       3D-array: (1, C, H, W)
                       2D-array: (1, 1, H, W)
                       1D-array: (1, 1, 1, W)

    Examples:
    >>> import numpy as np
    >>> x = np.arange(0, 2**12, dtype=int)
    >>> print(_get_shape(x))
    (1, 1, 1, 4096)
    >>> print(_get_shape(x.reshape(2**6, 2**6)))
    (1, 1, 64, 64)
    >>> print(_get_shape(x.reshape(2**4, 2**4, 2**4)))
    (1, 16, 16, 16)
    >>> print(_get_shape(x.reshape(2**3, 2**3, 2**3, 2**3)))
    (8, 8, 8, 8)
    """
    return (*(1,)*(4-array.ndim), *array.shape) if isinstance(array, ndarray) \
      else (*(1,)*(4-len(array)), *array)


def _get_stride(stride):
    """Get stride as tuple."""
    return stride if isinstance(stride, tuple) else (stride, stride)


def _get_pad(pad):
    """Get padding as tuple."""
    return pad if isinstance(pad, tuple) else (pad, pad)


def get_same_pad(I, F, stride):
    """
    Get padding width required to output the input shape as it is.

    Args:
        I (ndarray/tuple): Input array or input array's shape.
        F (ndarray/tuple): Filter array or filter array's shape.
        stride (tuple/int): Stride width.

    Returns:
        p_ud, p_lr (tuple): Padding width.

    Examples:
    >>> Ishape = (5, 5)
    >>> Fshape = (3, 3)
    >>> print(get_same_pad(Ishape, Fshape, 1))
    (1, 1)
    >>> print(get_same_pad(Ishape, Fshape, 2))
    (3, 3)
    """
    _, _, Ih, Iw = _get_shape(I)
    _, _, Fh, Fw = _get_shape(F)
    s_ud, s_lr = _get_stride(stride)

    def _calc(i, f, s):
        return int(np.ceil(0.5*((i-1)*s - i + f)))

    return _calc(Ih, Fh, s_ud), _calc(Iw, Fw, s_lr)


def get_Oshape(I, F, stride, pad):
    """
    Get padding width required to output the input shape as it is.

    Args:
        I (ndarray/tuple): Input array or input array's shape.
        F (ndarray/tuple): Filter array or filter array's shape.
        stride (tuple/int): Stride width.
        pad (tuple/int): Padding width.

    Returns:
        Oh, Ow (tuple): Output array's shape.

    Examples:
    >>> I = (5, 5)
    >>> F = (3, 3)
    >>> stride = 1
    >>> pad = 0
    >>> print(get_Oshape(I, F, stride, pad))
    (1, 3, 3)
    >>> print(get_Oshape(I, F, stride, get_same_pad(I, F, stride)))
    (1, 5, 5)
    """
    _, _, Ih, Iw = _get_shape(I)
    M, _, Fh, Fw = _get_shape(F)
    s_ud, s_lr = _get_stride(stride)
    p_ud, p_lr = _get_pad(pad)

    def _calc(i, f, s, p):
        return int((i - f + 2*p)//s + 1)

    return M, _calc(Ih, Fh, s_ud, p_ud), _calc(Iw, Fw, s_lr, p_lr)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
