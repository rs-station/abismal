import numpy as np
import gemmi
import tensorflow as tf
from abismal.distributions import Rice
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.layers import MLP
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.symmetry import ReciprocalASU,ReciprocalASUCollection,ReciprocalASUGraph
import reciprocalspaceship as rs
from tempfile import NamedTemporaryFile
from subprocess import call


class SpreadPosterior(StructureFactorPosteriorBase):
    def __init__(self, rac, Fc, sites_dict, wavelength_range, dmodel=32, mlp_depth=20, epsilon=1e-12, **kwargs):
        """
        rac : ReciprocalASUCollection
        mlp : tfk.layers.Layer
        Fc : np.array
        sites : dict{str : list[float] (3)}
        wavelength_range : list[float] (2)
        epsilon : float
        """
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.rac = rac
        self.dmodel = dmodel
        self.mlp_depth = mlp_depth
        self.Fc = Fc
        self.sites_dict = sites_dict
        self.wav_min, self.wav_max = wavelength_range
        self.num_atoms = len(sites_dict)

        self.sites = []
        for op in rac.reciprocal_asus[0].spacegroup.operations():
            self.sites.append([
                op.apply_to_xyz(v) for v in sites_dict.values()
            ])
        self.sites = tf.convert_to_tensor(self.sites)

        self.mlp = MLP(depth=mlp_depth, dmodel=dmodel)
        self.output_layer = tfk.layers.EinsumDense(
            '...d,dab->...ab',
            output_shape=(3, self.num_atoms),
            kernel_initializer='glorot_normal',
        )

    @property
    def cell(self):
        return self.rac.reciprocal_asus[0].cell

    @property
    def spacegroup(self):
        return self.rac.reciprocal_asus[0].spacegroup

    @staticmethod
    def estimate_wavelength_range(expt_files):
        from dxtbx.model import ExperimentList
        wavs = []
        for expt_file in expt_files:
            if not expt_file.endswith('.expt'):
                continue
            elist = ExperimentList.from_file(expt_file, check_format=False)
            for expt in elist:
                wavs.append(
                    expt.beam.get_wavelength()
                )
        return [np.min(wavs), np.max(wavs)]

    @staticmethod
    def sites_from_file(sites_pdb, elements):
        sites = {}
        structure = gemmi.read_pdb(sites_pdb)
        for model in structure:
            for chain in model:
                for resi in chain:
                    for atom in resi:
                        elem = atom.element.name
                        if elem in elements:
                            identifier = f"{model.num}/{chain.name}/{resi.seqid.num}/{atom.element.name}"
                            sites[identifier] = structure.cell.fractionalize(atom.pos).tolist() #Use the PDB's cell irrespective of rac
        return sites

    @staticmethod
    def reference_structure_factors(pdb_file, dmin, wavelength=None, energy=None):
        """
        Calculate anomalous structure factors from a model at a specific wavelength in Angstroms or energy in eV
        """
        if wavelength is None and energy is None:
            raise ValueError("Must specify either a wavelength in Angstroms or energy in eV")
        if wavelength is None:
            wavelength = rs.utils.ev2angstroms(energy)

        with NamedTemporaryFile(suffix='.mtz') as f:
            mtz_file = f.name
            command = f"gemmi sfcalc --to-mtz {mtz_file} --anomalous --dmin {dmin} --wavelength {wavelength} {pdb_file}"
            call(command.split())
            ds = rs.read_mtz(mtz_file)

        Fc = ds.to_structurefactor('FC', 'PHIC')
        Fa = ds.to_structurefactor('FCanom', 'PHICanom')

        out = ds[[]]
        Fplus = Fc + Fa
        Fminus = np.conjugate(Fc - Fa)
        out['F(+)'] = np.abs(Fplus)
        out['F(-)'] = np.abs(Fminus)
        out['PHI(+)'] = np.angle(Fplus, deg=True)
        out['PHI(-)'] = np.angle(Fminus, deg=True)
        out = out.infer_mtz_dtypes()
        return out

    @classmethod
    def from_pdb(cls, pdb_file, elements, dmin, wavelength_range=None, energy_range=None, standardize=True, **kwargs):
        """
        Build the spread posterior from a pdb file containing anomalous scatterers and an mtz file with "F(+/-)" and "PHI(+/-)" columns. 
        """
        valid = (wavelength_range is not None) ^ (energy_range is not None)
        if not valid:
            raise ValueError("Must specify either wavelength_range or energy_range")

        if wavelength_range is None:
            wavelength_range = [rs.utils.ev2angstroms(e) for e in energy_range]

        ds = SpreadPosterior.reference_structure_factors(pdb_file, dmin, wavelength_range[1])
        ds = ds.stack_anomalous()
        ds['Fcalc'] = ds.to_structurefactor('F', 'PHI')

        structure = gemmi.read_pdb(pdb_file)
        cell = structure.cell
        spacegroup = gemmi.SpaceGroup(structure.spacegroup_hm)

        sites = cls.sites_from_file(pdb_file, elements)
        rasu = ReciprocalASU(cell, spacegroup, dmin, anomalous=True)
        rac = ReciprocalASUGraph(rasu)
        Fc = rasu.gather(ds.Fcalc, rasu.Hunique)
        if standardize:
            Fc = Fc  / np.std(np.abs(Fc))

        return cls(rac, Fc, sites, wavelength_range, **kwargs)

    def scale_bijector(self, x):
        return tf.nn.softplus(x) + self.epsilon

    def distribution(self, params):
        loc, scale = tf.unstack(params)
        scale = self.scale_bijector(scale)
        q = tfd.Normal(loc, scale)
        return q

    def _distribution(self, loc, scale):
        q = tfd.Normal(
            loc, 
            scale, 
        )
        return q

    def distribution(self, asu_id, hkl, wav=None):
        if wav is None:
            wav = self.wav_min

        wav = (wav - self.wav_min) / (self.wav_max - self.wav_min)

        out = self.mlp(wav)
        out = self.output_layer(out) #moments
        fp,fpp,scale = tf.unstack(out, axis=-2)
        scale = self.scale_bijector(scale)

        # Rician RV params nu,sigma
        sigma = tf.math.sqrt(tf.einsum(
            "...s,osd->...",
            tf.square(scale),
            self.sites,
        ))

        h = tf.cast(hkl, 'float32')
        exp_arg = 2. * np.pi * tf.einsum(
            "...d,osd->...os",
            h,
            self.sites,
        )
        exponential = tf.complex(
            tf.math.cos(exp_arg),
            tf.math.sin(exp_arg),
        )
        f = tf.complex(fp, fpp)
        fc = self.rac.gather(self.Fc, asu_id, hkl)
        f = fc + tf.einsum("...s,...os->...", f, exponential)
        nu = tf.math.abs(f)

        q = Rice(nu, sigma)
        return q

    def flat_distribution(self=None, wav=0.):
        q = self._distribution(
            self.rac.asu_id,
            self.rac.Hunique,
            wav,
        )
        return q

    def sanitize_inputs(self, inputs):
        sane = []
        for x in inputs:
            if isinstance(x, tf.RaggedTensor):
                sane.append(x.flat_values)
            else:
                sane.append(x)
        return sane

    def call(self, inputs=None):
        (
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = self.sanitize_inputs(inputs)
        return self.distribution(asu_id, hkl_in, wavelength)

