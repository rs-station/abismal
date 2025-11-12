# Registry of optimizers for the abismal CLI
from abismal.optimizers.adam import Adam
from abismal.optimizers.adabelief import AdaBelief
from abismal.optimizers.wadam import WAdam

optimizer_dict = {
    'adam' : Adam,
    'adabelief' : AdaBelief,
    'wadam' : WAdam,
}
