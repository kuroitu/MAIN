import string
from typing import Any


def get_act(name: str, *args: Any, **kwds: Any) -> Any:
    """Interface for getting activation function.

    Args:
        name (str): Target activation function's name.

    Returns:
        _ (BaseAct): vation function class.

    Examples:
    >>> from main.dl.act import act_dict
    >>> for act in act_dict:
    ...     print(get_act(act))
    Step()
    Identity()
    BentIdentity()
    HardShrink(lambda_=0.5)
    SoftShrink(lambda_=0.5)
    Threshold(threshold=0, value=-1)
    Sigmoid()
    HardSigmoid()
    LogSigmoid()
    Tanh()
    TanhShrink()
    HardTanh()
    ReLU()
    ReLU6()
    LReLU(alpha=0.01)
    LReLU(alpha=0.01)
    ELU(alpha=1)
    SELU(lambda_=1.0507, alpha=1.67326)
    CELU(alpha=1)
    Softmax()
    Softmin()
    LogSoftmax()
    Softplus()
    Softsign()
    Swish(beta=1)
    Mish()
    TanhExp()
    ACON(acon_type='aconc', p1=1, p2=0.25, beta=ACON.__post_init__.<locals>._Beta(opt='adam', opt_dict={}, y=1), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), opt_dict={}, switch_dict={}, switch_opt='adam', switch_opt_dict={})
    ACON(acon_type='aconc', p1=1, p2=0.25, beta=ACON.__post_init__.<locals>._Beta(opt='adam', opt_dict={}, y=1), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), opt_dict={}, switch_dict={}, switch_opt='adam', switch_opt_dict={})
    ACON(acon_type='aconc', p1=1, p2=0.25, beta=ACON.__post_init__.<locals>._Beta(opt='adam', opt_dict={}, y=1), opt=Adam(alpha=0.001, beta1=0.9, beta2=0.999), opt_dict={}, switch_dict={}, switch_opt='adam', switch_opt_dict={})
    """
    from main.dl.act import act_dict

    name = name.lower().translate(str.maketrans("", "", string.punctuation))
    return act_dict[name](*args, **kwds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
