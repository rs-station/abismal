import numpy as np
import math
import gemmi
import tensorflow as tf
from abismal.distributions import Rice
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

class SpreadDistribution():
    def __init__(fc, fp, fpp, s):
        self.fc = fc
        self.fp = fp
        self.fpp = fpp
        self.s = s

    def sample(self, shape=()):
        zp = tfd.Normal(fp, s)
        zpp = tfd.Normal(fpp, s)
        return np.abs(self.fc + tf.complex(zp, zpp))



class SpreadGP(tfk.models.Model):
    def __init__(self, wavelength_range, num_atoms, num_points=100, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)
        epsilon = 1e-3
        small = 0.1
        self.num_atoms = num_atoms
        self.num_points = num_points
        self.epsilon = epsilon
        self.wav_min,self.wav_max = wavelength_range
        init = tfk.initializers.RandomUniform(0., 1.)
        self.inducing_x = self.add_weight('x', shape=(num_atoms, num_points), initializer=init)
        self.inducing_fp = self.add_weight('fp', shape=(num_atoms, num_points), initializer='zeros')
        self.inducing_fpp = self.add_weight('fpp', shape=(num_atoms, num_points), initializer='zeros')
        self.inducing_s = tfu.TransformedVariable(
            tf.eye(num_points, batch_shape=(num_atoms,)),
            tfb.Chain([
                tfb.CholeskyOuterProduct(),
                tfb.FillScaleTriL(
                    diag_bijector=tfb.Chain([
                        tfb.Shift(epsilon),
                        tfb.Exp(),
                    ])
                ),
            ]),
        )

        #self.jitter = tfu.TransformedVariable(
        #    small,
        #    tfb.Chain([
        #        tfb.Shift(epsilon),
        #        tfb.Exp(),
        #    ]),
        #)
        #self.jitter = epsilon
        self.jitter=0.
        self.bw = tfu.TransformedVariable(
            small,
            tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ]),
        )
        self.kfunc = tfm.psd_kernels.ExponentiatedQuadratic(
            length_scale = self.bw
        ) 

    def _get_variational_gps(self, wav):
        X = (wav - self.wav_min) / (self.wav_max - self.wav_min)
        qp = tfd.VariationalGaussianProcess(
              self.kfunc,
              X[None,None,...],
              self.inducing_x[None,...,None],
              self.inducing_fp[None,...],
              self.inducing_s[None,...],
              observation_noise_variance=0.,
              jitter=self.jitter,
        )
        qpp = tfd.VariationalGaussianProcess(
              self.kfunc,
              X[None,None,...],
              self.inducing_x[None,...,None],
              self.inducing_fpp[None,...],
              self.inducing_s[None,...],
              observation_noise_variance=0.,
              jitter=self.jitter,
        )
        return qp, qpp

    def call(self, wav):
        qp, qpp = self._get_variational_gps(wav)

        fp = tf.transpose(tf.squeeze(qp.mean(), axis=0))
        fpp = tf.transpose(tf.squeeze(qpp.mean(), axis=0))
        scale = tf.transpose(tf.squeeze(qp.stddev(), axis=0))
        self.add_metric(self.bw, "BW")
        self.add_metric(self.jitter, "Jitter")

#        kl_div = tf.reduce_mean(
#            qp.surrogate_posterior_kl_divergence_prior()
#        ) + tf.reduce_mean(
#            qpp.surrogate_posterior_kl_divergence_prior()
#        )
#
#        self.add_metric(kl_div, "KL")
#        self.add_loss(kl_div)

        return fp, fpp, scale

#    def compute_kl_terms(self, q, p, samples=None):
#        return None

class SpreadSmoother(tfk.models.Model):
    def __init__(self, wavelength_range, num_atoms, num_points=100, epsilon=1e-12, train_inducing_x=False, train_bw=False, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = 1e-3
        self.num_atoms = num_atoms
        self.num_points = num_points
        self.epsilon = epsilon
        self.wav_min,self.wav_max = wavelength_range
        if train_inducing_x:
            init = tfk.initializers.RandomUniform(self.wav_min, self.wav_max)
            self.inducing_x = self.add_weight('x', shape=(num_atoms, num_points), initializer=init, trainable=train_inducing_x)
        else:
            self.inducing_x = tf.linspace(self.wav_min, self.wav_max, num_points)[None,:]
        self.inducing_fp = self.add_weight('fp', shape=(num_atoms, num_points), initializer='zeros')
        self.inducing_fpp = self.add_weight('fpp', shape=(num_atoms, num_points), initializer='zeros')

        self.inducing_s = tfu.TransformedVariable(
            1e-3 * tf.ones((num_atoms, num_points)),
            tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ]),
        )
        #self.bw = 2. / num_points

        #bw = 2. * (self.wav_max - self.wav_min) / num_points,
        bw = (self.wav_max - self.wav_min) / 10.
        if train_bw:
            self.bw = tfu.TransformedVariable(
                bw,
                tfb.Chain([
                    tfb.Shift(epsilon),
                    tfb.Exp(),
                ]),
            )
        else:
            self.bw = bw
        self.b_factor = self.add_weight('b', shape=(num_atoms,), dtype='float32', initializer='zeros')

    def call(self, wav, dHKL=None):
        self.add_metric(self.bw, "BW")
        self.add_metric(tf.math.reduce_std(self.inducing_x), "SigX")
        d = tf.square((wav[:,None,...] - self.inducing_x[None,...]) / self.bw)
        w = tf.nn.softmax(-d, axis=-1)
        if dHKL is not None:
            T = tf.math.exp(-0.25 * self.b_factor[None,:,None] / tf.square(dHKL[:,None,None]))
            w = w * T
        fp = tf.einsum('...d,...d->...', self.inducing_fp, w)
        fpp = tf.einsum('...d,...d->...', self.inducing_fpp, w)
        scale  = tf.einsum('...d,...d->...', self.inducing_s, w)

        return fp, fpp, scale

    def compute_kl_terms(self, q, p, samples=None):
        qp = tfd.Normal(self.inducing_fp, self.inducing_s)
        qpp = tfd.Normal(self.inducing_fpp, self.inducing_s)
        p = tfd.Normal(0., 1.)

        kl_div = tf.reduce_mean(
            qp.kl_divergence(p) + qpp.kl_divergence(p)
        )
        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(kl_div, 'Custom_KL')

        return None

class SpreadNN(tfk.models.Model):
    def __init__(self, wavelength_range, num_atoms, epsilon=1e-12, dmodel=32, mlp_depth=5, activation='swish', gated=True, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = 1e-5
        self.num_atoms = num_atoms
        self.epsilon = epsilon
        self.wav_min,self.wav_max = wavelength_range
        self.input_layer = tfk.layers.Dense(dmodel, kernel_initializer='glorot_normal')
        self.mlp = MLP(depth=mlp_depth, activation=activation, gated=gated)
        self.output_layer = tfk.layers.EinsumDense(
            '...d,dab->...ab',
            output_shape=(3, self.num_atoms),
            kernel_initializer='glorot_normal',
            bias_axes='ab',
        )

    def scale_bijector(self, x):
        return tf.nn.softplus(x) + self.epsilon

    def encode_wav(self, wav):
        out = 2. * (wav - self.wav_min) / (self.wav_max - self.wav_min) - 1.
        f = 2. * np.pi * 2 ** tf.linspace(0., 5., 6)
        out = tf.concat((
            tf.math.cos(out * f),
            tf.math.sin(out * f),
        ), axis=-1)
        return out

    def call(self, wav):
        wav_normed = self.encode_wav(wav)
        out = self.input_layer(wav_normed)
        out = self.mlp(out)
        out = self.output_layer(out)
        fp,fpp,scale = tf.unstack(out, axis=-2)
        scale = self.scale_bijector(scale)

        qp = tfd.Normal(fp, scale)
        qpp = tfd.Normal(fpp, scale)
        p = tfd.Normal(0., 1.)

        kl_div = tf.reduce_mean(
            qp.kl_divergence(p) + qpp.kl_divergence(p)
        )
        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(kl_div, 'Custom_KL')
        return fp, fpp, scale

    def compute_kl_terms(self, q, p, samples=None):
        return None

class SpreadPosterior(StructureFactorPosteriorBase):
    def __init__(self, rac, Fc, Fmask, sites_dict, wavelength_range, spread_model=None, dmodel=32, epsilon=1e-12, optimize_fc=False, **kwargs):
        """
        rac : ReciprocalASUCollection
        Fc : np.array
        sites : dict{str : list[float] (3)}
        wavelength_range : list[float] (2)
        epsilon : float
        """
        super().__init__(rac, epsilon=epsilon, **kwargs)
        self.rac = rac
        self.dmodel = dmodel
        Fc = np.array(Fc)
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

        self.log_ksol = self.add_weight(
            name='ksol',
            shape=(),
            dtype='float32',
            initializer='zeros',
            trainable=True,
        )
        self.log_bsol = self.add_weight(
            name='bsol',
            shape=(),
            dtype='float32',
            initializer='zeros',
            trainable=True,
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
        self.sites = tf.convert_to_tensor(self.sites)

        if spread_model is None:
            #self.spread_model = SpreadNN(wavelength_range, self.num_atoms, dmodel=dmodel, epsilon=epsilon)
            #self.spread_model = SpreadGP(wavelength_range, self.num_atoms, dmodel,  epsilon)
            self.spread_model = SpreadSmoother(wavelength_range, self.num_atoms, 100,  epsilon)

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
    def ksol(self):
        return tf.math.exp(self.log_ksol)

    @property
    def bsol(self):
        return tf.math.exp(self.log_bsol)

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
        structure = gemmi.read_pdb(sites_pdb)
        for model in structure:
            for chain in model:
                for resi in chain:
                    for atom in resi:
                        elem = atom.element.name
                        if elem in elements:
                            identifier = f"{model.num}/{chain.name}/{resi.seqid.num}/{atom.name}"
                            if identifier in sites:
                                raise ValueError(f"Duplicate atom identifier in {sites_pdb}\nCannot proceed with analysis")
                            sites[identifier] = structure.cell.fractionalize(atom.pos).tolist() #Use the PDB's cell irrespective of rac
        return sites

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
            command = f"gemmi mask {pdb_file} {m.name}"
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
    def from_pdb(cls, pdb_file, elements, dmin, wavelength_range=None, energy_range=None, standardize=True, **kwargs):
        """
        Build the spread posterior from a pdb file containing anomalous scatterers and an mtz file with "F(+/-)" and "PHI(+/-)" columns. 
        """
        valid = (wavelength_range is not None) ^ (energy_range is not None)
        if not valid:
            raise ValueError("Must specify either wavelength_range or energy_range")

        if wavelength_range is None:
            wavelength_range = sorted([rs.utils.ev2angstroms(e) for e in energy_range])

        ds = SpreadPosterior.reference_structure_factors(pdb_file, dmin, wavelength_range[1], elements=elements)
        ds['Fcalc'] = ds.to_structurefactor('F', 'PHI')
        ds['Fmask'] = ds.to_structurefactor('Fmask', 'PHImask')

        structure = gemmi.read_pdb(pdb_file)
        cell = structure.cell
        spacegroup = gemmi.SpaceGroup(structure.spacegroup_hm)

        sites = cls.sites_from_file(pdb_file, elements)
        rasu = ReciprocalASU(cell, spacegroup, dmin, anomalous=True)
        rac = ReciprocalASUGraph(rasu)
        Fc = ds.loc[map(tuple, rasu.Hunique), 'Fcalc']
        Fmask = ds.loc[map(tuple, rasu.Hunique), 'Fmask']
        if standardize:
            Fc = Fc  #/ np.std(np.abs(Fc))
            Fmask = Fmask  #/ np.std(np.abs(Fmask))

        return cls(rac, Fc, Fmask, sites, wavelength_range, **kwargs)

    def distribution(self, asu_id, hkl, wav=None):
        if wav is None:
            wav = self.wav_min

        dhkl = self.rac.gather(self.rac.dHKL, asu_id, hkl)
        fp,fpp,scale = self.spread_model(wav, dhkl)

        # Rician RV params nu,sigma
        sigma = self.num_ops * tf.math.sqrt(tf.reduce_sum(tf.square(scale), axis=-1) + self.epsilon)

        #TODO -- just cache the phase?
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
        f = tf.complex(fp, fpp)
        fc = self.rac.gather(self.Fc, asu_id, hkl)
        fmask = self.rac.gather(self.Fmask, asu_id, hkl) 
        fmask = tf.complex(self.ksol * tf.math.exp(-self.bsol * tf.pow(dhkl, -2.)), 0.) * fmask

        f = fc + fmask + tf.einsum("...s,...os->...", f, exponential)
        nu = tf.math.abs(f)

        q = Rice(nu, sigma)
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
        return self.distribution(asu_id, hkl_in, wavelength)

    def compute_kl_terms(self, q, p, samples=None):
        if hasattr(self.spread_model, 'compute_kl_terms'):
            return self.spread_model.compute_kl_terms(q, p, samples)
        return super().compute_kl_terms(q, p, samples)

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

