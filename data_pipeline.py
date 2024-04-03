import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import librosa

from utils import one_hot_label, load_audio_waveform, dataset_from_csv

DATASET_DIR = "/data/fma_small/"
def get_features_from_waveform(audio):
    features = []
    # Convert TensorFlow tensor to NumPy array
    audio_np = audio.numpy()

    # Zero Crossing Rate
    zcr = librosa.zero_crossings(audio_np)
    features.append(np.sum(zcr))

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(audio_np)[0]
    features.append(np.mean(spectral_centroids))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(audio_np)
    features.append(np.mean(rolloff))

    # MFCCs (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(audio_np)

    for x in mfcc:
        features.append(np.mean(x))
        
    return tf.cast(features, tf.float32)

def get_dataset(input_csv, batch_size=8):
    """Function to build the dataset."""
    dataset = dataset_from_csv(input_csv)
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.strings.join([DATASET_DIR, sample["filename"]])))

    n_sample = 11025
    dataset = dataset.map(lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample, :]), num_parallel_calls=32)

    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample, 2))))

   # Apply get_features_from_waveform to each sample in the dataset
    dataset = dataset.map(lambda sample: dict(sample, features=get_features_from_waveform(sample["waveform"])))

    # Define the list of feature names
    features_names = ['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']

    # Map each sample to include its features with corresponding names
    dataset = dataset.map(lambda sample: dict(sample, **dict(zip(features_names, tf.unstack(sample["features"], axis=1, num=24)))))


    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map(lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))))

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