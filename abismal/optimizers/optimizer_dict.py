# Registry of optimizers for the abismal CLI
from abismal.optimizers.adam import Adam
from abismal.optimizers.adabelief import AdaBelief
import tf_keras as tfk

optimizer_dict = {
    'adam' : Adam,
    'adabelief' : AdaBelief,
    'tfkadam' : tfk.optimizers.Adam,
}
