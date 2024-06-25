"""
ABISMAL - merge serial diffraction data using neural networks and variational inference.
"""

import abismal.parser.io as io
import abismal.parser.architecture as architecture
import abismal.parser.likelihood as likelihood
import abismal.parser.optimizer as optimizer
import abismal.parser.phenix as phenix
import abismal.parser.priors as priors
import abismal.parser.surrogate_posterior as surrogate_posterior
import abismal.parser.tf as tf
import abismal.parser.training as training

groups = [
    architecture,
    io,
    likelihood,
    optimizer,
    phenix,
    priors,
    surrogate_posterior,
    tf,
    training,
]

from argparse import ArgumentParser
parser = ArgumentParser(description=__doc__)
for group in groups:
    g = parser.add_argument_group(group.title, group.description)
    for args,kwargs in group.args_and_kwargs:
        g.add_argument(*args, **kwargs)

all = [
    parser,
]
