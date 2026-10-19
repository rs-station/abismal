# Registry of optimizers for the abismal CLI
from abismal.optimizers.adam import Adam, AdamW, AdaBelief

optimizer_dict = {
    'adam' : Adam,
    'adamw' : AdamW,
    'adabelief' : AdaBelief,
}
