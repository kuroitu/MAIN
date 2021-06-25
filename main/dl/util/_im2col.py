import numpy as np

try:
    from ._util import _get_shape, _get_stride, _get_pad, get_Oshape
except ImportError:
    # For doctest
    from main.dl.util._util import (_get_shape, _get_stride, _get_pad,
                                    get_Oshape)


def im2col(images, filters, stride=1, pad=0):
    """
    Function that input nD-array converts to 2D-array.

    Args:
        images (ndarray): Target array.
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
        result (ndarray): Converted 2D-array -> (CFhFw, BOhOw)

    Examples:
    >>> import numpy as np
    >>> x = np.arange(0, 5*5).reshape(5, 5)
    >>> w = np.arange(0, 3*3).reshape(3, 3)
    >>> stride = 1
    >>> pad = 0
    >>> print(im2col(x, w, stride, pad))
    [[ 0.  1.  2.  5.  6.  7. 10. 11. 12.]
     [ 1.  2.  3.  6.  7.  8. 11. 12. 13.]
     [ 2.  3.  4.  7.  8.  9. 12. 13. 14.]
     [ 5.  6.  7. 10. 11. 12. 15. 16. 17.]
     [ 6.  7.  8. 11. 12. 13. 16. 17. 18.]
     [ 7.  8.  9. 12. 13. 14. 17. 18. 19.]
     [10. 11. 12. 15. 16. 17. 20. 21. 22.]
     [11. 12. 13. 16. 17. 18. 21. 22. 23.]
     [12. 13. 14. 17. 18. 19. 22. 23. 24.]]
    >>> from _util import get_same_pad
    >>> print(im2col(x, w, stride, get_same_pad(x, w, stride)))
    [[ 0.  0.  0.  0.  0.  0.  0.  1.  2.  3.  0.  5.  6.  7.  8.  0. 10. 11.
      12. 13.  0. 15. 16. 17. 18.]
     [ 0.  0.  0.  0.  0.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.
      13. 14. 15. 16. 17. 18. 19.]
     [ 0.  0.  0.  0.  0.  1.  2.  3.  4.  0.  6.  7.  8.  9.  0. 11. 12. 13.
      14.  0. 16. 17. 18. 19.  0.]
     [ 0.  0.  1.  2.  3.  0.  5.  6.  7.  8.  0. 10. 11. 12. 13.  0. 15. 16.
      17. 18.  0. 20. 21. 22. 23.]
     [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
      18. 19. 20. 21. 22. 23. 24.]
     [ 1.  2.  3.  4.  0.  6.  7.  8.  9.  0. 11. 12. 13. 14.  0. 16. 17. 18.
      19.  0. 21. 22. 23. 24.  0.]
     [ 0.  5.  6.  7.  8.  0. 10. 11. 12. 13.  0. 15. 16. 17. 18.  0. 20. 21.
      22. 23.  0.  0.  0.  0.  0.]
     [ 5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22.
      23. 24.  0.  0.  0.  0.  0.]
     [ 6.  7.  8.  9.  0. 11. 12. 13. 14.  0. 16. 17. 18. 19.  0. 21. 22. 23.
      24.  0.  0.  0.  0.  0.  0.]]
    """
    Ishape = B, C, Ih, Iw = _get_shape(images)
    # Filter's # channel must be same as input one, so discard.
    Fshape = _, _, Fh, Fw = _get_shape(filters)
    stride = s_ud, s_lr = _get_stride(stride)
    pad = p_ud, p_lr = _get_pad(pad)
    _, Oh, Ow = get_Oshape(Ishape, Fshape, stride, pad)
    p0 = (0, 0)

    images = np.pad(images.reshape(*Ishape),
                    [p0, p0, (p_ud, p_ud), (p_lr, p_lr)],
                    "constant")
    result = np.empty((B, C, Fh, Fw, Oh, Ow))
    for h in range(Fh):
        hlim = h + s_ud*Oh
        for w in range(Fw):
            wlim = w + s_lr*Ow
            result[:, :, h, w, :, :] = images[:, :, h:hlim:s_ud, w:wlim:s_lr]
    return result.transpose(1, 2, 3, 0, 4, 5).reshape(C*Fh*Fw, B*Oh*Ow)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
