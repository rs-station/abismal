# Registry of optimizers for the abismal CLI
from tf_keras.optimizers import Adam
from abismal.optimizers.adabelief import AdaBelief
from abismal.optimizers.wadam import WAdam
from abismal.optimizers.lazy_adam import LazyAdam,LazyAdaBelief

optimizer_dict = {
    'adam' : Adam,
    'adabelief' : AdaBelief,
    'wadam' : WAdam,
    'lazyadam' : LazyAdam,
    'lazyadabelief' : LazyAdaBelief,
}
