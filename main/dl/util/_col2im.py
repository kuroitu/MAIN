import numpy as np
from numpy import ndarray

try:
    from ._util import _get_shape, _get_stride, _get_pad, get_Oshape
except ImportError:
    # For doctest
    from main.dl.util._util import (_get_shape, _get_stride, _get_pad,
                                    get_Oshape)


def col2im(cols, images, filters, stride=1, pad=0):
    """
    Function that input nD-array converts to 2D-array.

    Args:
        cols (ndarray): Target array.
                        2D-array -> (CFhFw, BOhOw)
        images (ndarray/tuple): Target array or array's shape.
                                4D-array -> (B, C, Ih, Iw)
                                3D-array -> (C, Ih, Iw)
                                2D-array -> (Ih, Iw)
                                1D-array -> (Iw)
        filters (ndarray/tuple): Filter array or array's shape.
                                 4D-array -> (M, C, Fh, Fw)
                                 3D-array -> (C, Fh, Fw)
                                 2D-array -> (Fh, Fw)
                                 1D-array -> (Fw)
        stride (tuple/int): Stride width.
                            tuple -> (ud, lr)
                            int -> stride
        pad (tuple/int): Padding width.
                             tuple -> (ud, lr)
                             int -> pad

    Returns:
        images (ndarray): Converted nD-array
                          (B, C, Ih, Iw) -> images.shape

    Examples:
    >>> import numpy as np
    >>> from _im2col import im2col
    >>> x = np.arange(0, 5*5).reshape(5, 5)
    >>> w = np.arange(0, 3*3).reshape(3, 3)
    >>> stride = 1
    >>> pad = 0
    >>> print(cols:=im2col(x, w, stride, pad))
    [[ 0.  1.  2.  5.  6.  7. 10. 11. 12.]
     [ 1.  2.  3.  6.  7.  8. 11. 12. 13.]
     [ 2.  3.  4.  7.  8.  9. 12. 13. 14.]
     [ 5.  6.  7. 10. 11. 12. 15. 16. 17.]
     [ 6.  7.  8. 11. 12. 13. 16. 17. 18.]
     [ 7.  8.  9. 12. 13. 14. 17. 18. 19.]
     [10. 11. 12. 15. 16. 17. 20. 21. 22.]
     [11. 12. 13. 16. 17. 18. 21. 22. 23.]
     [12. 13. 14. 17. 18. 19. 22. 23. 24.]]
    >>> print(col2im(cols, x, w, stride, pad))
    [[  0.   2.   6.   6.   4.]
     [ 10.  24.  42.  32.  18.]
     [ 30.  66. 108.  78.  42.]
     [ 30.  64. 102.  72.  38.]
     [ 20.  42.  66.  46.  24.]]
    """
    Ishape = B, C, Ih, Iw = _get_shape(images)
    # Filter's # channel must be same as input one, so discard.
    Fshape = _, _, Fh, Fw = _get_shape(filters)
    stride = s_ud, s_lr = _get_stride(stride)
    pad = p_ud, p_lr = _get_pad(pad)
    _, Oh, Ow = get_Oshape(Ishape, Fshape, stride, pad)

    cols = cols.reshape(C, Fh, Fw, B, Oh, Ow).transpose(3, 0, 1, 2, 4, 5)
    result = np.zeros((B, C, Ih + 2*p_ud + s_ud - 1, Iw + 2*p_lr + s_lr - 1))
    for h in range(Fh):
        hlim = h + s_ud*Oh
        for w in range(Fw):
            wlim = w + s_lr*Ow
            result[:, :, h:hlim:s_ud, w:wlim:s_lr] += cols[:, :, h, w, :, :]
    if isinstance(images, ndarray):
        return result[:, :, p_ud : Ih+p_ud, p_lr : Iw+p_lr] \
                .reshape(*images.shape)
    elif isinstance(images, tuple):
        return result[:, :, p_ud : Ih+p_ud, p_lr : Iw+p_lr] \
                .reshape(*images)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
