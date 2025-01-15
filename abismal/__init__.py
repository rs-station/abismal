## Version number for abismal
def getVersionNumber():
    version = None
    try:
        from setuptools.version import metadata

        version = metadata.version("abismal")
    except ImportError:
        from setuptools.version import pkg_resources

        version = pkg_resources.require("abismal")[0].version

    return version

__version__ = getVersionNumber() 

import pandas as pd #Workaround for scipy 1.15.1 (see https://github.com/rs-station/abismal/issues/12)
from abismal.ragged import quiet
from abismal import callbacks,distributions,io,layers,likelihood,merging,optimizers,scaling,surrogate_posterior,symmetry
