import tensorflow as tf
import tf_keras as tfk
from os.path import exists,dirname,abspath
import reciprocalspaceship as rs
from os import mkdir
import numpy as np

class MtzSaver(tfk.callbacks.Callback):
    def __init__(self, output_directory, reference_mtz=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_mtz = reference_mtz
        if self.reference_mtz is not None:
            self.reference_mtz = rs.read_mtz(reference_mtz)
        self.output_directory = abspath(output_directory)

        if not exists(self.output_directory):
            mkdir(output_directory)

    def reindex_dataset(self, merged, anomalous):
        keys = merged.keys()
        if anomalous:
            merged = merged.stack_anomalous()
        def get_first_key_of_dtype(ds, dtype):
            keys = ds.dtypes[ds.dtypes == dtype].keys()
            if len(keys) > 0:
                return keys[0]
            return None

        def calculate_correlation(R, F, SigF=None):
            if SigF is not None:
                ds = R.join(F).join(SigF).dropna()
                x,y,w = ds.to_numpy().T
                w = np.reciprocal(np.square(w))
                return rs.utils.weighted_pearsonr(x, y, w)
            ds = R.join(F).dropna()
            return ds.corr(method='pearson').iloc[0,1]
            
        ref = self.reference_mtz
        refkey = get_first_key_of_dtype(ref, 'F')
        if refkey is None:
            return merged

        best = -1.
        mtz_out = None
        for op in self.model.reindexing_ops:
            op = op.gemmi_op #These are abismal ops. we need gemmi
            reindexed = merged.apply_symop(op).hkl_to_asu(anomalous=anomalous)
            cc = calculate_correlation(ref[[refkey]], reindexed[['F']], reindexed[['SIGF']])
            if cc > best:
                best = cc
                mtz_out = reindexed
        if anomalous:
            mtz_out = mtz_out.unstack_anomalous()
        mtz_out = mtz_out[keys] #reorder columns to match input spec
        return mtz_out

    def on_epoch_end(self, epoch, logs=None):
        self.save_mtz(epoch)

    def save_mtz(self, epoch, seen=True):
        for asu_id,data in enumerate(self.model.surrogate_posterior.to_datasets(seen=seen)):
            if self.reference_mtz is not None:
                anomalous = self.model.surrogate_posterior.rac.reciprocal_asus[asu_id].anomalous
                data = self.reindex_dataset(data, anomalous)
            data.write_mtz(f"{self.output_directory}/asu_{asu_id}_epoch_{epoch+1}.mtz")

class FriedelMtzSaver(MtzSaver):
    """
    Save friedelized inputs into a single mtz.
    """
    column_names = (
        'F(+)',
        'SIGF(+)',
        'F(-)',
        'SIGF(-)',
        # There's a bug with anomalous E-values
        #'E(+)', 
        #'SIGE(+)',
        #'E(-)',
        #'SIGE(-)',
        'I(+)',
        'SIGI(+)',
        'I(-)',
        'SIGI(-)',
    )
    def save_mtz(self, epoch, seen=True):
        ds_plus,ds_minus = self.model.surrogate_posterior.to_datasets(seen=seen)
        data = rs.concat((
            ds_plus,
            ds_minus.apply_symop('-x,-y,-z'),
        )).unstack_anomalous()
        data = data[[k for k in self.column_names if k in data]]
        data.write_mtz(f"{self.output_directory}/asu_0_epoch_{epoch+1}.mtz")


