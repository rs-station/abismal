
def list_of_ints(string):
    return [int(i) for i in string.split(',')]

def list_of_floats(string):
    return [float(i) for i in string.split(',')]

def list_of_ops(string):
    return string.split(';')

def float_or_none(string):
    """
    Parse a float, or the literal "none".

    Several options take a float whose absence selects different behaviour --
    --studentt-dof picks a normal error model, --ff-epsilon falls back to
    --epsilon. Those used to be reachable by leaving the option unset, but both
    now default to a value, so "none" has to be spellable on the command line.
    """
    if string.strip().lower() == "none":
        return None
    return float(string)
