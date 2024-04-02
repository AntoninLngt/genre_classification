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
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.strings.join([DATASET_DIR, sample["filename"]])))
    
    # Use the new function to load and preprocess audio
    dataset = dataset.map(lambda sample: (load_and_preprocess_audio(sample["filename"])))
    
    # Filter out badly shaped waveforms if necessary (make sure to define check_valid)
    # dataset = dataset.filter(lambda sample: check_valid(sample))
    
    # Encode labels
    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map(lambda waveform, mfccs, spectrogram, genre: (waveform, mfccs, spectrogram, one_hot_label(genre, tf.constant(label_list))))
    
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