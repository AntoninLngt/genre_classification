import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import one_hot_label, load_audio_waveform, dataset_from_csv,load_and_preprocess_audio

DATASET_DIR = "/data/fma_small/"

def get_dataset(input_csv, batch_size=8):
    """Function to build the dataset."""
    dataset = dataset_from_csv(input_csv)
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.string_join([DATASET_DIR, sample["filename"]])))
    
    # Use the new function to load and preprocess audio
    waveform, mfccs, spectrogram=load_and_preprocess_audio(sample["filename"])
    dataset = dataset.map(lambda sample: dict(sample, waveform, mfccs, spectrogram))
    
    # Filter out badly shaped waveforms
    dataset = dataset.filter(lambda sample: check_valid(sample))
    
    # Encode labels
    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map(lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))))
    
    # Select relevant features
    dataset = dataset.map(lambda sample: (sample["waveform"], sample["mfccs"], sample["spectrogram"], sample["one_hot_label"]))
    
    # Create batches
    dataset = dataset.batch(batch_size)
    
    return dataset


# test dataset data generation
if __name__=="__main__":

    dataset = get_dataset("fma_small.csv")
    batch = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:

        # Evaluate first batch
        batch_value = sess.run(batch)
        print("Training dataset generated a batch with:")
        for el in batch_value:
            print(f"A {type(el)} with shape {el.shape}.")