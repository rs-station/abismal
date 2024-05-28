import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers  as tfl
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as tfk
from abismal.symmetry import ReciprocalASU,ReciprocalASUCollection
from abismal.merging import VariationalMergingModel
from abismal.surrogate_posterior.structure_factor import WilsonPosterior,PosteriorCollection
from abismal.callbacks import HistorySaver,MtzSaver
from abismal.io import StillsLoader
from abismal.scaling import ImageScaler
from IPython import embed



# Handle GPU selection
def set_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(gpus)
    if gpus:
        try:
            if gpu_id is None:
                tf.config.experimental.set_visible_devices([], 'GPU')
            else:
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

debug = False
eager = False

# Load filenames from text
expt_files = [i for i in open('expt_files.txt').read().strip().split('\n') if i[0]!="#"]
refl_files = [i for i in open('refl_files.txt').read().strip().split('\n') if i[0]!="#"]

# Hardware Parameters
gpu_id = 1
set_gpu(gpu_id)

# Data Parameters
outdir = "results/less_kl"
if os.path.exists(outdir):
    answer = input(f"""Permission to overwrite output directory {outdir} (y/N)""") + ' '
    if answer[0].lower()=='y':
        from shutil import rmtree
        rmtree(outdir)
    else:
        from sys import exit
        exit()

weights_file = None # f"results/merge/weights.tf"
test_size = 5_000 #test set image count
mc_samples=50 
epochs=100
batch_size=1
shuffle_buffer_size=100_000
dmin = 1.8
anomalous = True

# model dimensions
model_dims = 32
ff_dims = 2 * model_dims
num_blocks = 20

# likelihood stuff
studentt_dof = None

# kullback leibler divergences / regularizers 
kl_weight = 1e-8
scale_kl_weight = 10.
dropout=None

# Optimizer settings
#learning_rate=1e-3
boundaries = [50_000]
values = [1e-3, 1e-4]
learning_rate = tfk.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-12
global_clipnorm=None #Adam gradient clipping
clipnorm=None #Adam gradient clipping per parameter
clipvalue=None  #Adam gradient clipping
amsgrad=False #Whether to use amsgrad


# layer options
activation='relu'
kernel_initializer=tfk.initializers.VarianceScaling(scale=1./10., mode='fan_avg', distribution='truncated_normal', seed=1234)

# Write out this script for logging purposes
if not os.path.exists(outdir):
    os.mkdir(outdir)
with open(outdir + "/merge.py", "w") as out:
    out.write(open(__file__).read())

loader = StillsLoader(expt_files, refl_files, dmin=dmin)

rasu = ReciprocalASU(
    loader.cell,
    loader.spacegroup,
    dmin,
    anomalous=anomalous,
)
rac = ReciprocalASUCollection(rasu)

surrogate_posterior = PosteriorCollection(
    WilsonPosterior(rasu, kl_weight, eps=epsilon),
)

from abismal.scaling.scaling import DeltaFunctionLayer
scale_posterior = DeltaFunctionLayer(kernel_initializer=kernel_initializer)

scale_model = ImageScaler(
    mlp_width=model_dims, 
    mlp_depth=num_blocks, 
    dropout=dropout,
    hidden_units=ff_dims,
    layer_norm=False,
    activation=activation,
    kernel_initializer=kernel_initializer,
    kl_weight=scale_kl_weight,
    eps=epsilon,
    scale_posterior=scale_posterior,
)

model = VariationalMergingModel(
    scale_model, 
    surrogate_posterior, 
    mc_samples=mc_samples,
    studentt_dof=studentt_dof, 
)

data = loader.get_dataset()


opt = tfk.optimizers.Adam(
    learning_rate, 
    beta_1, 
    beta_2, 
    global_clipnorm=global_clipnorm, 
    clipnorm=clipnorm, 
    clipvalue=clipvalue, 
    epsilon=epsilon, 
)


train = data.skip(test_size)
test = data.take(test_size)

steps_per_epoch = 10_000 
validation_steps = 1_000 
train = train.cache().repeat()
test = test.cache().repeat()

if shuffle_buffer_size is not None:
    train = train.shuffle(shuffle_buffer_size)

train = train.batch(batch_size)
test = test.batch(batch_size)

#Debugging
if debug:
    for x,y in test:
        break
    out = model(x)
    from IPython import embed
    embed(colors='linux')
    from sys import exit
    exit()

mtz_saver = MtzSaver(outdir)
history_saver = HistorySaver(outdir)
weight_saver  = tf.keras.callbacks.ModelCheckpoint(filepath=f'{outdir}/weights.tf', save_weights_only=True, verbose=1)

callbacks = [
    mtz_saver,
    history_saver,
    weight_saver,
]

model.compile(opt, run_eagerly=eager)
if weights_file is not None:
    model.load_weights(weights_file)
    model.optimizer.learning_rate.assign(learning_rate)

train,test = train.prefetch(tf.data.AUTOTUNE),test.prefetch(tf.data.AUTOTUNE)

history = model.fit(x=train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=callbacks, validation_data=test)

embed(colors='linux')

