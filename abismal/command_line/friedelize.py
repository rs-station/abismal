"""
Split various reflection files into two files -- one for each Friedel mate.
Multiple files can be processed with the same call. .mtz, .stream, and .refl are all supported. 

Example:
The following will take the input stream file and create two new files (cxidb_61_plus.stream and cxidb_61_minus.stream).
abismal.friedelize cxidb_61.stream
"""


import numpy as np
from argparse import ArgumentParser
import reciprocalspaceship as rs
from reciprocalspaceship.decorators import spacegroupify
import gemmi

@spacegroupify
def is_friedel_plus(hkls, spacegroup):
    _,isym = rs.utils.hkl_to_asu(hkls, spacegroup)
    fplus = (isym % 2) == 1
    return fplus

def friedelize_mtz(file, spacegroup=None):
    mtz = rs.read_mtz(file)
    base = file.removesuffix('.mtz')
    if spacegroup is not None:
        mtz.spacegroup = spacegroup
    fplus = is_friedel_plus(mtz.get_hkls(), mtz.spacegroup)
    mtz[fplus].write_mtz(base + '_plus.mtz')
    mtz[~fplus].write_mtz(base + '_minus.mtz')

def friedelize_stream(file, spacegroup):
    refl_start = "Reflections measured after indexing"
    refl_end = "End of reflections"
    if spacegroup is None:
        spacegroup = gemmi.SpaceGroup(1)

    base = file.removesuffix('.stream')
    plus = base + '_plus.stream'
    minus = base + '_minus.stream'
    in_refl_list = False
    with iter(open(file)) as f, open(plus, 'w') as p, open(minus, 'w') as m:
        for line in f:
            if line.startswith(refl_start):
                in_refl_list = True
                p.write(line);m.write(line)
                line = next(f)
                p.write(line);m.write(line)
                continue
            if line.startswith(refl_end):
                in_refl_list = False
            if not in_refl_list:
                p.write(line);m.write(line)
                continue
            hkl = [int(i) for i in line.split()[:3]]
            is_plus = np.squeeze(is_friedel_plus([hkl], spacegroup))
            if is_plus:
                p.write(line)
            else:
                m.write(line)

def friedelize_refl(file, spacegroup):
    extension = file.split('.')[-1]
    base = file.removesuffix('.' + extension)
    from dials.array_family import flex
    refls = flex.reflection_table.from_file(file)
    hkl = np.array(refls['miller_index'])
    plus = is_friedel_plus(hkl, spacegroup)
    refls.select(flex.bool(plus)).as_file(base + '_plus.' + extension)
    refls.select(flex.bool(~plus)).as_file(base + '_minus.' + extension)

def friedelize(file, spacegroup='P 1'):
    if file.endswith(".stream"):
        friedelize_stream(file, spacegroup)
    elif file.endswith(".pickle") or file.endswith(".refl"):
        friedelize_refl(file, spacegroup)
    elif file.endswith(".mtz"):
        friedelize_mtz(file, spacegroup)
    else:
        raise ValueError(f"Cannot determine filetype for input file {file}")

def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("input_file", nargs='+')
    parser.add_argument("--spacegroup", "-s", help="The space group of these data. If not supplied, attempt to infer it from the file falling back to P1", default=None)

    parser = parser.parse_args()
    for file in parser.input_file:
        friedelize(file, parser.spacegroup)

