import tensorflow as tf


def split_dataset_train_test(data, test_frac, seed=1234):
    """ Deterministically split data into fractions """
    train = data.enumerate().filter(
        lambda i,x: tf.random.stateless_uniform([1], (i+seed, i+seed))[0] > test_frac
    ).map(
        lambda i,x: x
    )
    test  = data.enumerate().filter(
        lambda i,x: tf.random.stateless_uniform([1], (i+seed, i+seed))[0] <= test_frac
    ).map(
        lambda i,x: x
    )
    return train, test

