import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf
import librosa

from utils import one_hot_label, load_audio_waveform, dataset_from_csv

DATASET_DIR = "/data/fma_small/"

def audio_pipeline(audio):
    features = []

    # Evaluate the TensorFlow tensor within a TensorFlow session
    with tf.Session() as sess:
        audio_np = sess.run(audio)

    # Calcul du ZCR
    zcr = librosa.zero_crossings(audio_np)
    features.append(np.sum(zcr))

    # Calcul de la moyenne du Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(audio_np)[0]
    features.append(np.mean(spectral_centroids))
  
    # Calcul du spectral rolloff point
    rolloff = librosa.feature.spectral_rolloff(audio_np)
    features.append(np.mean(rolloff))

    # Calcul des moyennes des MFCC
    mfcc = librosa.feature.mfcc(audio_np)

    for x in mfcc:
        features.append(np.mean(x))

    return features


def get_dataset(input_csv, batch_size=8):
    """Function to build the dataset."""
    dataset = dataset_from_csv(input_csv)
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.strings.join([DATASET_DIR, sample["filename"]])))

    n_sample = 11025
    dataset = dataset.map(lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample, :]), num_parallel_calls=32)

    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample, 2))))

    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map(lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))))

    dataset = dataset.map(lambda sample: (sample["waveform"],
                                        audio_pipeline(sample["waveform"]),
                                        sample["one_hot_label"]))

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