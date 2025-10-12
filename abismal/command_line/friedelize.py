"""
Split various reflection files into two files -- one for each Friedel mate.
Multiple files can be processed with the same call. .mtz, .stream, and .refl are all supported. 
Centrics are retained in the "_plus" file. 

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
    centric = rs.utils.is_centric(hkls, spacegroup)
    return fplus | centric

def friedelize_mtz(file_name, spacegroup=None):
    mtz = rs.read_mtz(file_name)
    base = file_name.removesuffix('.mtz')
    if spacegroup is not None:
        mtz.spacegroup = spacegroup
    fplus = is_friedel_plus(mtz.get_hkls(), mtz.spacegroup)
    mtz[fplus].write_mtz(base + '_plus.mtz')
    mtz[~fplus].write_mtz(base + '_minus.mtz')

@spacegroupify
def friedelize_stream(file_name, spacegroup=1):
    refl_start = "Reflections measured after indexing"
    refl_end = "End of reflections"

    go = spacegroup.operations()
    rasu = gemmi.ReciprocalAsu(spacegroup)

    base = file_name.removesuffix('.stream')
    plus = base + '_plus.stream'
    minus = base + '_minus.stream'
    in_refl_list = False
    with iter(open(file_name)) as f, open(plus, 'w') as p, open(minus, 'w') as m:
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
            _,isym = rasu.to_asu(hkl, go)
            fplus = (isym % 2) == 1
            centric = go.centric_flag_array([hkl])[0]
            is_plus = fplus | centric
            if is_plus:
                p.write(line)
            else:
                m.write(line)

def friedelize_refl(file_name, spacegroup):
    extension = file_name.split('.')[-1]
    base = file_name.removesuffix('.' + extension)
    from dials.array_family import flex
    refls = flex.reflection_table.from_file(file_name)
    hkl = np.array(refls['miller_index'])
    plus = is_friedel_plus(hkl, spacegroup)
    refls.select(flex.bool(plus)).as_file(base + '_plus.' + extension)
    refls.select(flex.bool(~plus)).as_file(base + '_minus.' + extension)

def friedelize(file_name, spacegroup='P 1'):
    if file_name.endswith(".stream"):
        friedelize_stream(file_name, spacegroup)
    elif file_name.endswith(".pickle") or file_name.endswith(".refl"):
        friedelize_refl(file_name, spacegroup)
    elif file_name.endswith(".mtz"):
        friedelize_mtz(file_name, spacegroup)
    else:
        raise ValueError(f"Cannot determine file for input file {file_name}")

def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("input_file", nargs='+')
    parser.add_argument("--spacegroup", "-s", help="The space group of these data. If not supplied, attempt to infer it from the file falling back to P1", default=None)

    parser = parser.parse_args()
    for file_name in parser.input_file:
        friedelize(file_name, parser.spacegroup)
