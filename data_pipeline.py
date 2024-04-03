import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import one_hot_label, load_audio_waveform, dataset_from_csv

DATASET_DIR = "/data/fma_small/"
def audio_pipeline(audio, fs=44100):
    # Compute zero crossings
    # Determine sign changes
    sign_changes = tf.abs(tf.sign(audio) - tf.sign(audio[:, :-1]))
    # Count the number of sign changes
    zcr = tf.reduce_mean(sign_changes, axis=1)

    # Compute STFT
    stft = tf.contrib.signal.stft(audio, frame_length=256, frame_step=128, fft_length=256)

    # Compute the magnitude spectrum
    magnitude_spectrum = tf.abs(stft)

    # Define frequency bins
    frequencies = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=tf.shape(magnitude_spectrum)[-1],
        sample_rate=fs,
        lower_edge_hertz=0.0,
        upper_edge_hertz=fs / 2)  # Nyquist frequency

    # Compute the spectral centroid
    centroid = tf.tensordot(magnitude_spectrum, frequencies, 1)

    # Compute the magnitude spectrum
    magnitude = tf.abs(stft)

    # Compute the cumulative sum along the frequency axis
    cumulative_sum = tf.cumsum(magnitude, axis=1)

    # Find the frequency index where the cumulative sum exceeds 85% of the total energy
    total_energy = tf.reduce_sum(magnitude, axis=1)
    threshold = 0.85 * total_energy[:, None]
    index = tf.argmax(tf.cast(cumulative_sum >= threshold, tf.int32), axis=1)

    # Compute log power Mel spectrogram manually
    linear_to_mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=tf.shape(magnitude_spectrum)[-1],
        sample_rate=fs,
        lower_edge_hertz=0.0,
        upper_edge_hertz=fs / 2)  # Nyquist frequency

    mel_spectrogram = tf.tensordot(magnitude_spectrum, linear_to_mel_matrix, 1)
    log_mel_spectrogram = tf.log(mel_spectrogram + 1e-6)  # Add a small value to avoid log(0)

    # Compute MFCCs from the log Mel spectrogram
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrogram,
        num_mfccs=20
    )


    # Stack all features
    features = tf.concat([tf.expand_dims(zcr, axis=1), centroid,mfccs], axis=1)
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