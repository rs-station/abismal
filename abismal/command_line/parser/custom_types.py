
def list_of_ints(string):
    return [int(i) for i in string.split(',')]

def list_of_floats(string):
    return [float(i) for i in string.split(',')]

def list_of_ops(string):
    return string.split(';')

def transform_name(string):
    """
    Validate a name for --activation / --pre-activation.

    Both slots accept either a normalizer (a key of ``FeedForward.norm_dict``) or
    any keras activation, so validation has to consult both. Checking here rather
    than with argparse's `choices` keeps the keras half open-ended -- keras gains
    activations, and listing a frozen set would go stale.
    """
    import argparse

    from abismal.layers import FeedForward
    import tf_keras as tfk

    if string in FeedForward.norm_dict:
        return string
    try:
        tfk.activations.get(string)
    except (ValueError, KeyError, TypeError):
        normalizers = ", ".join(sorted(FeedForward.norm_dict))
        raise argparse.ArgumentTypeError(
            f"{string!r} is neither a normalizer ({normalizers}) nor a keras activation"
        )
    return string
