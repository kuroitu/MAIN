try:
    from ._base import BaseBlock
except ImportError:
    # For doctest
    from main.dl.layer._base import BaseBlock


class Middle(BaseBlock):
    """Middle layer class.
    All implemenetations are already implemented in 'BaseBlock' now.
    Input layer is also treated as one of middle layers in mounting.
    """
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
