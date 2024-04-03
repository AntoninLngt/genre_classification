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

    # Compute zero crossings
    zcr = tf.py_func(librosa.zero_crossings, [audio], tf.bool)
    features.append(sum(zcr))

    # Compute spectral centroid
    spectral_centroids = tf.py_func(librosa.feature.spectral_centroid, [audio], tf.float32)[0]
    features.append(np.mean(spectral_centroids))

    # Compute spectral rolloff
    rolloff = tf.py_func(librosa.feature.spectral_rolloff, [audio], tf.float32)[0]
    features.append(np.mean(rolloff))

    # Compute MFCCs
    mfccs = tf.py_func(librosa.feature.mfcc, [audio], tf.float32)
    for mfcc in mfccs:
        features.append(np.mean(mfcc))

    return features

def audio_pipeline(audio):
    features = []

    # Compute zero crossings
    zcr = tf.py_func(librosa.zero_crossings, [audio[:, 0]], tf.bool)
    features.append(tf.cast(tf.reduce_sum(tf.cast(zcr, tf.float32)), tf.float32))

    # Compute spectral centroid
    spectral_centroids = tf.py_func(librosa.feature.spectral_centroid, [audio[:, 0]], tf.float32)[0]
    features.append(tf.reduce_mean(spectral_centroids))

    # Compute spectral rolloff
    rolloff = tf.py_func(librosa.feature.spectral_rolloff, [audio[:, 0]], tf.float32)[0]
    features.append(tf.reduce_mean(rolloff))

    # Compute MFCCs
    mfccs = tf.py_func(librosa.feature.mfcc, [audio[:, 0]], tf.float32)
    for mfcc in mfccs:
        features.append(tf.reduce_mean(mfcc))

    return features


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