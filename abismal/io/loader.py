#/usr/bin/env cctbx.python
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import tensorflow as tf


class DataLoader():
    data_labels = (
        "ASU",
        "HKL",
        "Resolution",
        "Wavelength",
        "Metadata",
        "I",
        "SigI",
    ) #To support laue add wavelength and resolution here
    def __init__(self, metadata_length):
        self.signature=(
            (
                tf.RaggedTensorSpec((None, None, 1), tf.int32, 1, tf.int32), #ASU ID
                tf.RaggedTensorSpec((None, None, 3), tf.int32, 1, tf.int32), #HKL 
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #dHKL
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #wavelength
                tf.RaggedTensorSpec((None, None, metadata_length), tf.float32, 1, tf.int32), #metadata
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #I
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #SigI
            ),(
                tf.RaggedTensorSpec((None, None, 1), tf.float32, 1, tf.int32), #I 
            )
        )

