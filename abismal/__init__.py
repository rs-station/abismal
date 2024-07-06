# Version number for reciprocalspaceship
def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("abismal")[0].version
    return version


__version__ = getVersionNumber()

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = environ.get("TF_CPP_MIN_LOG_LEVEL", "3")

from abismal import callbacks,distributions,io,layers,likelihood,merging,optimizers,scaling,surrogate_posterior,symmetry
