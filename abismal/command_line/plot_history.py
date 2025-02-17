"""
Plot the history file from an Abismal merging result
"""

import pandas as pd
import numpy as np
from pylab import plt
import seaborn as sns
from argparse import ArgumentParser

default_keys = [
    "Epoch",
    "Time (s)",
#    "FB Total (MiB)",
#    "FB Reserved (MiB)",
#    "FB Used (MiB)",
#    "FB Free (MiB)",
#    "BAR1 Total (MiB)",
#    "BAR1 Used (MiB)",
#    "BAR1 Free (MiB)",
#    "Conf Compute Protected Total (MiB)",
#    "Conf Compute Protected Used (MiB)",
#    "Conf Compute Protected Free (MiB)",
    "loss",
#    "Istd",
#    "Icount",
    "NLL",
    "KL",
    "wCCpred",
    "CCpred",
#    "Image",
#    "ImageStd",
#    "Σ_loc",
#    "Σ_scale",
    "KL_Σ",
#    "Σ_mean",
#    "Σ_std",
    "|∇s|",
    "|∇q|",
#    "val_loss",
#    "val_Istd",
#    "val_Icount",
#    "val_NLL",
#    "val_KL",
#    "val_wCCpred",
#    "val_CCpred",
#    "val_Image",
#    "val_ImageStd",
#    "val_Σ_loc",
#    "val_Σ_scale",
#    "val_KL_Σ",
#    "val_Σ_mean",
#    "val_Σ_std"
]

def run(parser):
    keys = parser.keys
    if keys is None:
        keys = default_keys
    df = pd.read_csv(parser.csv_file)
    for k in keys:
        val_keys = []
        if f'val_{k}' in df:
            val_keys.append(f'val_{k}')
        keys = keys + val_keys

    #Filter by keys
    df = df[keys] 

    #Make data 'tidy' for seaborn
    data = df.melt("Epoch")
    data['Set'] = np.array(['Train', 'Test'])[data['variable'].str.startswith('val_').to_numpy('int')]
    data['variable'] = data['variable'].str.removeprefix('val_')


    sns.lineplot(
        data,
        x='Epoch',
        y='value',
        hue='variable',
        style='Set',
        palette='Dark2',
    )
    plt.semilogy()
    plt.grid(which='both', axis='both', ls='-.')
    plt.show()

def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("csv_file", help="A history csv file from an abismal run.")
    parser.add_argument("--keys", nargs='+', default=None, help="Keys you want to plot.")
    parser = parser.parse_args()
    run(parser)


