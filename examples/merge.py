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
from abismal.merging import VariationalMergingModel,PosteriorCollection,WilsonPosterior
from abismal.callbacks import HistorySaver,MtzSaver
from abismal.io import StillsLoader
from abismal.scaling import ImageScaler
from IPython import embed


def breakpoint():
    from IPython import embed
    embed(colors='linux')
    from sys import exit
    exit()


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
#debug = True
eager = False
#eager = True

# Load filenames from text
expt_files = [i for i in open('expt_files.txt').read().strip().split('\n') if i[0]!="#"]
refl_files = [i for i in open('refl_files.txt').read().strip().split('\n') if i[0]!="#"]
first_run_only = False
#first_run_only = True

# Hardware Parameters
gpu_id = 0
set_gpu(gpu_id)

# Data Parameters
outdir = "results/scratch"
if os.path.exists(outdir):
    answer = input(f"""Permission to overwrite output directory {outdir} (y/N)""") + ' '
    if answer[0].lower()=='y':
        from shutil import rmtree
        rmtree(outdir)
    else:
        from sys import exit
        exit()

weights_file = None 
#weights_file = f"results/simple_kl/weights.tf"
test_size = 5_000
mc_samples=50
epochs=100
batch_size=100
dmin = 1.8
anomalous = True


# Model Hyperparameters
# model dimensions
model_dims = 32
ff_dims = 2 * model_dims
num_blocks = 50

# kullback leibler divergences / regularizers 
kl_weight = 1e-3
dropout=None

# Optimizer settings
#learning_rate=1e-4
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

#kernel_initializer=tfk.initializers.VarianceScaling(scale=1./5./max_reflections_per_image, mode='fan_avg', distribution='truncated_normal')
#kernel_initializer=FanMaxInitializer(scale=1. / num_blocks / 5.)
#kernel_initializer=tfk.initializers.VarianceScaling(scale=1./2, mode='fan_avg', distribution='truncated_normal')
kernel_initializer=tfk.initializers.VarianceScaling(scale=1./10., mode='fan_avg', distribution='truncated_normal', seed=1234)

# likelihood stuff
studentt_dof = None


# Just cache the data in a temporary directory
from tempfile import TemporaryDirectory
datadir = TemporaryDirectory()

# Write out this script for logging purposes
if not os.path.exists(outdir):
    os.mkdir(outdir)
with open(outdir + "/merge.py", "w") as out:
    out.write(open(__file__).read())



if first_run_only:
    expt_files = [expt_files[0]]
    refl_files = [refl_files[0]]

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

scale_model = ImageScaler(
    mlp_width=model_dims, 
    mlp_depth=num_blocks, 
    dropout=dropout,
    hidden_units=ff_dims,
    layer_norm=False,
    activation=activation,
    kernel_initializer=kernel_initializer,
)

model = VariationalMergingModel(scale_model, surrogate_posterior, 
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


data = data.cache()
train = data.skip(test_size)
test = data.take(test_size)
train,test = train.batch(batch_size),test.batch(batch_size)

if first_run_only:
    phenix_frequency = 5
    steps_per_epoch = None
    validation_steps = None
else:
    phenix_frequency = 10
    steps_per_epoch = int(100_000 / batch_size)
    validation_steps = int(5_000 / batch_size)
    train,test = train.repeat(),test.repeat()


#Debugging
if debug:
    batched = data.batch(2)
    inputs = next(batched.as_numpy_iterator())[0]
    out = model(inputs)
    breakpoint()

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

