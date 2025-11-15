"""
ABISMAL - merge serial diffraction data using neural networks and variational inference.
"""

import abismal.command_line.parser.io as io
import abismal.command_line.parser.architecture as architecture
import abismal.command_line.parser.likelihood as likelihood
import abismal.command_line.parser.optimizer as optimizer
import abismal.command_line.parser.phenix as phenix
import abismal.command_line.parser.priors as priors
import abismal.command_line.parser.ray as ray
import abismal.command_line.parser.surrogate_posterior as surrogate_posterior
import abismal.command_line.parser.tf as tf
import abismal.command_line.parser.training as training

groups = {
    'Architecture' : architecture,
    'IO' : io,
    'Likelihood' : likelihood,
    'Optimizer' : optimizer,
    'PHENIX' : phenix,
    'Priors' : priors,
    'Ray' : ray,
    'Surrogate Posterior' : surrogate_posterior,
    'TensorFlow' : tf,
    'Training' : training,
}

from argparse import ArgumentParser
parser = ArgumentParser(description=__doc__)
for group in groups.values():
    g = parser.add_argument_group(group.title, group.description)
    for args,kwargs in group.args_and_kwargs:
        g.add_argument(*args, **kwargs)

all = [
    parser,
]
