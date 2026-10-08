import numpy as np


_MASK64 = (1 << 64) - 1


def splitmix64(x):
    """The splitmix64 finalizer: a well mixed 64-bit hash of a 64-bit int."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def index_uniform(index, seed):
    """A uniform number in [0, 1) that depends only on `(index, seed)`.

    This is what makes per-image choices, such as train/test membership,
    reproducible no matter the order or thread they are made in.
    """
    h = splitmix64((splitmix64(int(seed) & _MASK64) ^ int(index)) & _MASK64)
    # The top 53 bits fill a double's mantissa exactly.
    return (h >> 11) * (1.0 / (1 << 53))


def is_test_index(index, test_frac, seed=1234):
    """Whether element `index` of the dataset belongs in the test split.

    Membership is a fixed function of the element's position in the input and
    the seed, so the same files give the same split on every run. It does not
    reproduce the split the TensorFlow version of abismal drew.
    """
    if test_frac <= 0.:
        return False
    return index_uniform(index, seed) <= test_frac


def split_dataset_train_test(data, test_frac, seed=1234):
    """ Deterministically split an iterable of images into (train, test) lists """
    train, test = [], []
    for i, x in enumerate(data):
        (test if is_test_index(i, test_frac, seed) else train).append(x)
    return train, test
