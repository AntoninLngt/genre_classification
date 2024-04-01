import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import one_hot_label, load_audio_waveform, dataset_from_csv

DATASET_DIR = "/data/fma_small/"

def get_dataset(input_csv, batch_size=8):

    # build dataset from csv file
    dataset = dataset_from_csv(input_csv)

    # add directory in the filename
    dataset = dataset.map( lambda sample: dict(sample, filename=tf.string_join([DATASET_DIR, sample["filename"]])))


    n_sample = 11025
    # load audio and take first quarter of second only
    dataset = dataset.map( lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample,:]), num_parallel_calls=32)

    # Filter out badly shaped waveforms (due to loading errors)
    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample,2))))

    # one hot encoding of labels
    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map( lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))) )

    # Select only features and annotation
    dataset = dataset.map(lambda sample: (sample["waveform"], sample["one_hot_label"]))

    # Make batch
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