import numpy as np
import math
import gemmi
import tensorflow as tf
from abismal.distributions import Rice,FoldedNormal
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk
from abismal.layers import MLP
from abismal.surrogate_posterior import StructureFactorPosteriorBase
from abismal.symmetry import ReciprocalASU,ReciprocalASUCollection,ReciprocalASUGraph
import reciprocalspaceship as rs
from tempfile import NamedTemporaryFile
from subprocess import call
from abismal.surrogate_posterior.spread.gp import SpreadGP
from abismal.surrogate_posterior.spread.nn import SpreadNN
from abismal.surrogate_posterior.spread.smoother import SpreadSmoother

def wav_min_max(expt_file):
    from dxtbx.model import ExperimentList
    wav_min = math.inf
    wav_max = -math.inf
    if expt_file.endswith('.expt'):
        elist = ExperimentList.from_file(expt_file, check_format=False)
        for expt in elist:
            wav = expt.beam.get_wavelength()
            wav_min = min(wav, wav_min)
            wav_max = max(wav, wav_max)
    return [wav_min, wav_max]

class SpreadPosterior(StructureFactorPosteriorBase):
    def __init__(self, rac, Fc, Fmask, sites_dict, wavelength_range, element_name, charge=0, spread_model=None, dmodel=32, epsilon=1e-12, optimize_fc=False, atomic_b_init=20., bsol_init=0.5, ksol_init=1.0, kl_weight=1e-2, standardize=True, **kwargs):
        """
        rac : ReciprocalASUCollection
        Fc : np.array
        sites : dict{str : list[float] (3)}
        wavelength_range : list[float] (2)
        epsilon : float
        """
        super().__init__(rac, epsilon=epsilon, **kwargs)
        std = 1.

        self.rac = rac
        self.dmodel = dmodel
        cell = self.rac.reciprocal_asus[0].cell 

        dHKL = self.rac.dHKL
        rd2 = np.power(dHKL, -2.)
        self.element_name = element_name
        if charge is not None:
            gemmi.IT92_set_ignore_charge(False)
            coeff = gemmi.IT92_get_exact(gemmi.Element(element_name), charge)  # for Mg2+
        self.f0 = np.array(list(map(coeff.calculate_sf, 0.25 * rd2)), 'float32')

        Fc = np.array(Fc)
        if standardize: 
            std = np.abs(Fc).std()
            Fc = Fc / std
            Fmask = Fmask / std
            self.f0 = self.f0 / std

        self.logFabs = self.add_weight(
            name='logFabs',
            shape=Fc.shape,
            dtype='float32',
            initializer=tfk.initializers.Constant(tf.math.log(tf.math.abs(Fc))),
            trainable=optimize_fc,
        )
        self.Phi = self.add_weight(
            name='Phi',
            shape=Fc.shape,
            dtype='float32',
            initializer=tfk.initializers.Constant(tf.math.angle(Fc)),
            trainable=False,
        )
        self.Fmask_abs = self.add_weight(
            name='Fmask',
            shape=Fc.shape,
            dtype='float32',
            initializer=tfk.initializers.Constant(tf.math.log(tf.math.abs(Fmask))),
            trainable=False,
        )
        self.PHImask = self.add_weight(
            name='PHImask',
            shape=Fc.shape,
            dtype='float32',
            initializer=tfk.initializers.Constant(tf.math.angle(Fmask)),
            trainable=False,
        )

        self.sites_dict = sites_dict
        self.wav_min, self.wav_max = wavelength_range
        self.num_atoms = len(sites_dict)

        self.sites = []
        self.num_ops = 0
        for op in rac.reciprocal_asus[0].spacegroup.operations():
            self.sites.append([
                op.apply_to_xyz(v) for v in sites_dict.values()
            ])
            self.num_ops += 1

        sites = tf.convert_to_tensor(self.sites)
        self.sites = self.add_weight(
            name='sites',
            shape=(self.num_ops, self.num_atoms, 3),
            dtype='float32',
            initializer=tfk.initializers.Constant(sites),
            trainable=False,
        )

        if spread_model is None:
            #self.spread_model = SpreadNN(wavelength_range, self.num_atoms, dmodel=dmodel, epsilon=epsilon)
            #self.spread_model = SpreadGP(wavelength_range, self.num_atoms, dmodel,  epsilon)
            self.spread_model = SpreadSmoother(wavelength_range, self.num_atoms, 100,  epsilon, kl_weight=kl_weight)

        self._ksol = self.add_weight(
            name='_ksol',
            shape=(),
            dtype='float32',
            initializer=tfk.initializers.Constant(tf.math.log(ksol_init)),
            trainable=True,
        )
        self._bsol = self.add_weight(
            name='bsol',
            shape=(),
            dtype='float32',
            initializer=tfk.initializers.Constant(tf.math.log(bsol_init)),
            trainable=True,
        )

        self._k = self.add_weight(
            name='k',
            shape=(),
            dtype='float32',
            initializer='zeros',
            trainable=True,
        )

        log_b_init = tf.math.log(atomic_b_init)
        self._atomic_b_factor = self.add_weight(
            'atomic_b', 
            shape=(self.num_atoms,), 
            dtype='float32', 
            initializer=tfk.initializers.Constant(log_b_init),
            trainable=False,
        )
        self._b = self.add_weight(
            'b', 
            shape=(), 
            dtype='float32', 
            initializer='zeros'
        )

    def get_config(self):
        config = super().get_config()
        config['rac'] = tfk.saving.serialize_keras_object(self.rac)
        config['Freal'] = tfk.saving.serialize_keras_object(tf.math.real(self.Fc))
        config['Fimag'] = tfk.saving.serialize_keras_object(tf.math.imag(self.Fc))
        config['sites_dict'] = self.sites_dict
        config['dmodel'] = self.dmodel 
        config['epsilon'] = self.epsilon
        config['wavelength_range'] = (self.wav_min, self.wav_max)
        return config

    @classmethod
    def from_config(cls, config):
        config['rac'] = tfk.saving.deserialize_keras_object(config['rac'])
        config['Fc'] = tf.complex(
            tfk.saving.deserialize_keras_object(config.pop('Freal')),
            tfk.saving.deserialize_keras_object(config.pop('Fimag')),
        )
        return cls(**config)


    @property
    def k(self):
        return tf.math.exp(self._k)

    @property
    def b(self):
        #return self._b
        return -tf.math.exp(self._b)

    @property
    def ksol(self):
        return tf.math.exp(self._ksol)

    @property
    def bsol(self):
        return tf.math.exp(self._bsol)

    @property
    def atomic_b_factor(self):
        return tf.math.exp(self._atomic_b_factor)

    @property
    def Fc(self):
        Fabs = tf.math.exp(self.logFabs)
        return  tf.complex(
            Fabs * tf.math.cos(self.Phi),
            Fabs * tf.math.sin(self.Phi),
        )

    @property
    def Fmask(self):
        return  tf.complex(
            self.Fmask_abs * tf.math.cos(self.PHImask),
            self.Fmask_abs * tf.math.sin(self.PHImask),
        )

    @property
    def cell(self):
        return self.rac.reciprocal_asus[0].cell

    @property
    def spacegroup(self):
        return self.rac.reciprocal_asus[0].spacegroup

    @staticmethod
    def estimate_wavelength_range(expt_files, num_cpus=1):
        from tqdm import tqdm
        from joblib import Parallel,delayed

        wav_min = math.inf
        wav_max = -math.inf

        if num_cpus == 1:
            results = map(wav_min_max, expt_files)
        else:
            results = Parallel(num_cpus)(delayed(wav_min_max)(efile) for efile in expt_files)

        for _wav_min,_wav_max in tqdm(results, total=len(expt_files)):
            wav_min = min(wav_min, _wav_min)
            wav_max = max(wav_max, _wav_max)

        return [wav_min, wav_max]

    @staticmethod
    def sites_from_file(sites_pdb, elements):
        sites = {}
        b_factors = {}
        structure = gemmi.read_pdb(sites_pdb)
        for model in structure:
            for chain in model:
                for resi in chain:
                    for atom in resi:
                        elem = atom.element.name
                        if elem in elements:
                            identifier = f"{model.num}/{chain.name}/{resi.seqid.num}/{atom.name}"
                            b_factors[identifier] = atom.b_iso
                            if identifier in sites:
                                raise ValueError(f"Duplicate atom identifier in {sites_pdb}\nCannot proceed with analysis")
                            sites[identifier] = structure.cell.fractionalize(atom.pos).tolist() #Use the PDB's cell irrespective of rac
        return sites, b_factors

    @staticmethod
    def remove_sites_from_file(sites_pdb, output_pdb, elements):
        sites = {}
        structure = gemmi.read_pdb(sites_pdb)
        resis = []
        to_delete = []
        # Deleting atoms during iteration will cause baddd things to happen. 
        # Need to iterate first to figure out all the atoms to be deleted
        for model in structure:
            for chain in model:
                for resi in chain:
                    for atom in resi:
                        identifier = f"{model.num}/{chain.name}/{resi.seqid.num}/{atom.name}"
                        elem = atom.element.name
                        if elem in elements:
                            resis.append(resi)
                            to_delete.append((
                                atom.name, atom.altloc
                            ))
        for resi,args in zip(resis, to_delete):
            resi.remove_atom(*args)
        structure.write_pdb(output_pdb)

    def fc_to_dataset(self):
        h,k,l = self.rac.Hunique.numpy().T
        ds = rs.DataSet({
            'H' : h,
            'K' : k,
            'L' : l,
            'F-model' : np.abs(self.Fc),
            'PHIF-model' : np.angle(self.Fc, deg=True),
        }, cell=self.cell, spacegroup=self.spacegroup, merged=True).infer_mtz_dtypes()
        return ds

    @staticmethod
    def reference_structure_factors(pdb_file, dmin, wavelength=None, energy=None, elements=None):
        """
        Calculate anomalous structure factors from a model at a specific wavelength in Angstroms or energy in eV
        """
        if wavelength is None and energy is None:
            raise ValueError("Must specify either a wavelength in Angstroms or energy in eV")
        if wavelength is None:
            wavelength = rs.utils.ev2angstroms(energy)

        with NamedTemporaryFile(suffix='.mtz') as f, NamedTemporaryFile(suffix='.pdb') as p:
            mtz_file = f.name
            pdb_in = p.name
            if elements is not None:
                SpreadPosterior.remove_sites_from_file(pdb_file, pdb_in, elements)
            else:
                pdb_in = pdb_file

            command = f"gemmi sfcalc --ksolv=0 --to-mtz {mtz_file} --anomalous --dmin {dmin} --wavelength {wavelength} {pdb_in}"
            call(command.split())
            ds = rs.read_mtz(mtz_file)

        with NamedTemporaryFile(suffix='.mtz') as f, NamedTemporaryFile(suffix='.msk') as m:
            #Note the spacing needs to be specified to match oversampling of the atoms' fft
            command = f"gemmi mask --spacing={dmin / 3.0} {pdb_file} {m.name}"
            call(command.split())
            command = f"gemmi map2sf {m.name} {f.name} Fmask PHImask --dmin {dmin}"
            call(command.split())
            mask = rs.read_mtz(f.name)

        Fc = ds.to_structurefactor('FC', 'PHIC')
        Fa = ds.to_structurefactor('FCanom', 'PHICanom')

        Fplus = Fc + Fa
        Fminus = np.conjugate(Fc - Fa)
        Fminus = Fminus.reset_index().rename(columns={0 : 'F'})
        Fminus.loc[:,["H", "K", "L"]] = -Fminus.loc[:,["H", "K", "L"]]
        out = rs.concat((
            Fplus.reset_index().rename(columns={0 : 'F'}),
            Fminus,
        ), check_isomorphous=False,)
        out = out.set_index(['H', 'K', 'L'])
        out = rs.concat(out.from_structurefactor('F'), axis=1)
        out = out.rename(columns={'Phi': 'PHI'})
        out.spacegroup = ds.spacegroup
        out.cell = ds.cell
        idx = out.hkl_to_asu().get_hkls()
        out['Fmask'] = mask['Fmask'].loc[map(tuple, idx)].to_numpy()
        out['PHImask'] = mask['PHImask'].loc[map(tuple, idx)].to_numpy()
        out = out.infer_mtz_dtypes()
        return out

    @classmethod
    def from_pdb(cls, pdb_file, element, dmin, charge=0, wavelength_range=None, energy_range=None, standardize=False, **kwargs):
        """
        Build the spread posterior from a pdb file containing anomalous scatterers and an mtz file with "F(+/-)" and "PHI(+/-)" columns. 
        """
        if wavelength_range is None and energy_range is None:
            raise ValueError("Must specify either wavelength_range or energy_range")
        if wavelength_range is None:
            wavelength_range = sorted([rs.utils.ev2angstroms(e) for e in energy_range])

        ds = SpreadPosterior.reference_structure_factors(pdb_file, dmin, wavelength_range[1], elements=[element])
        ds['Fcalc'] = ds.to_structurefactor('F', 'PHI')
        ds['Fmask'] = ds.to_structurefactor('Fmask', 'PHImask')

        structure = gemmi.read_pdb(pdb_file)
        cell = structure.cell
        spacegroup = gemmi.SpaceGroup(structure.spacegroup_hm)

        sites,b_factors = cls.sites_from_file(pdb_file, [element])
        b_factors = np.array(list(b_factors.values()), dtype='float32')
        rasu = ReciprocalASU(cell, spacegroup, dmin, anomalous=True)
        rac = ReciprocalASUGraph(rasu)
        Fc = ds.loc[map(tuple, rasu.Hunique), 'Fcalc']
        Fmask = ds.loc[map(tuple, rasu.Hunique), 'Fmask']
        if standardize:
            scale = np.std(np.abs(Fc))
            Fc = Fc / scale
            Fmask = Fmask / scale

        return cls(rac, Fc, Fmask, sites, wavelength_range, element_name=element, charge=charge, atomic_b_init=b_factors, **kwargs)

    def distribution(self, asu_id, hkl, wav=None):
        if wav is None:
            wav = self.wav_min

        dhkl = self.rac.gather(self.rac.dHKL, asu_id, hkl)
        invd2 = tf.pow(dhkl, -2.)

        #Scale corrections
        k = tf.complex(self.k, 0.)
        T = tf.complex(
            tf.math.exp(-self.b * invd2),
            0.,
        )
        ksol = tf.complex(self.ksol, 0.)
        Tsol = tf.complex(
            tf.math.exp(-self.bsol * invd2),
            0.,
        )

        # Complex scattering factors
        fp,fpp,scale = self.spread_model(wav, dhkl)
        f0 = self.rac.gather(self.f0, asu_id, hkl)

        # Atomic b-factor correction
        Ta = tf.math.exp(-0.25 * self.atomic_b_factor[None,:] * invd2[:,None])
        fp = Ta * fp
        fpp = Ta * fpp
        f0 = f0[:,None] * Ta
        scale = Ta * scale

        # Rician RV params nu,sigma
        sigma = tf.math.sqrt(self.num_ops * tf.reduce_sum(tf.square(scale), axis=-1) + self.epsilon)

        #TODO -- just cache the phase and use gather
        h = tf.cast(hkl, 'float32')
        phase = 2. * np.pi * tf.einsum(
            "...d,osd->...os",
            h,
            self.sites,
        )
        exponential = tf.complex(
            tf.math.cos(phase),
            tf.math.sin(phase),
        )
        fs = tf.complex(f0 + fp, fpp)
        fs = tf.einsum("...s,...os->...", fs, exponential)
        fc = self.rac.gather(self.Fc, asu_id, hkl)
        fmask = ksol * Tsol * self.rac.gather(self.Fmask, asu_id, hkl) 

        f = k * T * (fc + fmask + fs)
        nu = tf.math.abs(f)

        q = Rice(nu, sigma)
        #q = FoldedNormal(nu, sigma, self.epsilon)
        return q

    def flat_distribution(self, wav=None):
        if wav is None:
            wav = self.wav_min
        wav = wav * tf.ones_like(self.rac.asu_id[:,None], dtype='float32')
        q = self.distribution(
            self.rac.asu_id[:,None],
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

    def get_results(self, npoints=100):
        import pandas as pd
        wav = tf.linspace(self.wav_min, self.wav_max, npoints)[:,None]
        fp,fpp,scale = self.spread_model(wav)
        wav = wav * tf.ones_like(fp)
        atom = tf.ones_like(scale, dtype='int32') * tf.range(scale.shape[-1])
        results = pd.DataFrame({
            "wavelength" : wav.numpy().flatten(),
            "f'" : fp.numpy().flatten(),
            "f''" : fpp.numpy().flatten(),
            "stddev" : scale.numpy().flatten(),
            "atom_id" : atom.numpy().flatten(),
        })
        results['atom_name'] = np.array(list(self.sites_dict.keys()))[atom.numpy().flatten()]
        return results

    def call(self, inputs=None):
        ( asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = self.sanitize_inputs(inputs)
        self.add_metric(self.ksol, 'ksol')
        self.add_metric(self.bsol, 'bsol')
        self.add_metric(self.k, 'k')
        self.add_metric(self.b, 'b')
        self.add_metric(tf.math.reduce_mean(self.atomic_b_factor), "atomic_b_ave")
        return self.distribution(asu_id, hkl_in, wavelength)

class DummySpreadPosterior(SpreadPosterior):
    def __init__(self, rac, Fc, sites_dict, wavelength_range, spread_model=None, dmodel=32, mlp_depth=20, epsilon=1e-12, snr=100., **kwargs):
        """
        rac : ReciprocalASUCollection
        Fc : np.array
        sites : dict{str : list[float] (3)}
        wavelength_range : list[float] (2)
        epsilon : float
        """
        super().__init__(
            rac, Fc, sites_dict, wavelength_range, spread_model=None, dmodel=32, mlp_depth=20, epsilon=1e-12, **kwargs)
        self.spread_model = None
        self.snr = snr

    def distribution(self, asu_id, hkl, wav=None):
        fc = self.rac.gather(self.Fc, asu_id, hkl)
        nu = tf.math.abs(fc)
        sigma = nu / self.snr
        q = Rice(nu, sigma)
        return q

    def get_results(self, npoints=100):
        import pandas as pd
        results = pd.DataFrame()
        return results

