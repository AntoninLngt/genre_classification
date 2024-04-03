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
    # Compute zero crossings
    zcr = tf.cast(tf.abs(tf.sign(audio[:, :-1] * audio[:, 1:])), tf.float32)
    zcr = tf.reduce_sum(zcr, axis=-1)

    # Compute STFT of the audio waveform
    stft = tf.signal.stft(audio_frame, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)

    # Compute spectral centroid
    magnitude_spectrum = tf.abs(stft)
    frequencies = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=128, num_spectrogram_bins=1025, sample_rate=sample_rate, lower_edge_hertz=0.0, upper_edge_hertz=8000.0)
    spectral_centroids = tf.tensordot(magnitude_spectrum, frequencies, 1)
    spectral_centroids = tf.reduce_sum(spectral_centroids, axis=1)
    # Compute spectral rolloff
    cumsum = tf.cumsum(stft, axis=1)
    total_energy = tf.reduce_sum(stft, axis=1)
    rolloff = tf.argmax(tf.where(cumsum >= 0.85 * total_energy[:, None], True, False), axis=1)

    # Compute MFCCs
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(tf.math.log(stft + 1e-6))
    mfccs = tf.reduce_mean(mfccs, axis=1)

    # Stack all features
    features = tf.stack([zcr, spectral_centroids, tf.cast(rolloff, tf.float32)] + [mfccs], axis=1)
    return features

def get_dataset(input_csv, batch_size=8):
    """Function to build the dataset."""
    dataset = dataset_from_csv(input_csv)
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.strings.join([DATASET_DIR, sample["filename"]])))

    n_sample = 11025
    dataset = dataset.map(lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample, :]), num_parallel_calls=32)

    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample, 2))))

   # Apply get_features_from_waveform to each sample in the dataset
    dataset = dataset.map(lambda sample: dict(sample, features=audio_pipeline(sample["waveform"])))

    # Define the list of feature names
    features_names = ['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']

    # Map each sample to include its features with corresponding names
    dataset = dataset.map(lambda sample: dict(sample, **dict(zip(features_names, tf.unstack(sample["features"], axis=1,num=23)))))


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