# Registry of optimizers for the abismal CLI
from abismal.optimizers.adam import Adam
from abismal.optimizers.adabelief import AdaBelief
from abismal.optimizers.wadam import WAdam
import tf_keras as tfk

optimizer_dict = {
    'adam' : Adam,
    'adabelief' : AdaBelief,
    'wadam' : WAdam,
    'tfkadam' : tfk.optimizers.Adam,
}
