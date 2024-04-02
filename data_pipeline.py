import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import one_hot_label, load_audio_waveform, dataset_from_csv,mfccs,spectrogram

DATASET_DIR = "/data/fma_small/"

def get_dataset(input_csv, batch_size=8):
    """Function to build the dataset."""
    dataset = dataset_from_csv(input_csv)
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.strings.join([DATASET_DIR, sample["filename"]])))

    n_sample = 11025
    dataset = dataset.map(lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample, :]), num_parallel_calls=32)

    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample, 2))))

    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map(lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))))

    dataset = dataset.map(lambda sample: (sample["waveform"], mfccs(sample["filename"]), spectrogram(sample["filename"]), sample["one_hot_label"]))

    dataset = dataset.batch(batch_size)

    return dataset


# test dataset data generation
if __name__=="__main__":

    dataset = get_dataset("fma_small.csv")
    # Create an iterator
    iterator = iter(dataset)

    try:
        # Get the first batch
        batch_value = next(iterator)
        print("Training dataset generated a batch with:")
        for el in batch_value:
            print(f"A {type(el)} with shape {el.shape}.")
    except tf.errors.OutOfRangeError:
        print("End of dataset reached.")