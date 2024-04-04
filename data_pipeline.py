import json
import time
import os
from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import one_hot_label, load_audio_waveform, dataset_from_csv

DATASET_DIR = "/data/fma_small/"
def zcr(audio):
    # Determine sign changes
    sign_changes = tf.abs(tf.sign(audio) - tf.sign(audio[:, :-1]))
    # Count the number of sign changes
    zcr = tf.reduce_mean(tf.cast(sign_changes, tf.float32), axis=1)
    return zcr

def centroid(audio, fs=44100):
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
        upper_edge_hertz=fs / 2
    )
    # Compute the spectral centroid
    centroid = tf.tensordot(magnitude_spectrum, frequencies, 1)
    return centroid

def mfcc(audio, fs=44100):
    print(audio)
    # Compute STFT
    stft = tf.contrib.signal.stft(audio, frame_length=256, frame_step=128, fft_length=256)
    # Compute the magnitude spectrum
    magnitude_spectrum = tf.abs(stft)
    # Compute the Mel spectrogram
    linear_to_mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=20,
        num_spectrogram_bins=tf.shape(magnitude_spectrum)[-1],
        sample_rate=fs,
        lower_edge_hertz=0.0,
        upper_edge_hertz=fs / 2
    )
    mel_spectrogram = tf.tensordot(magnitude_spectrum, linear_to_mel_matrix, 1)
    log_mel_spectrogram = tf.log(mel_spectrogram + 1e-6)  # Log-compressed Mel spectrogram
    # Compute MFCCs from the log Mel spectrogram
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    return mfccs

def get_dataset(input_csv, batch_size=8):
    """Function to build the dataset."""
    dataset = dataset_from_csv(input_csv)
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.strings.join([DATASET_DIR, sample["filename"]])))

    n_sample = 11025
    # load audio and take first quarter of second only
    dataset = dataset.map( lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample,:]), num_parallel_calls=32)

    # Filter out badly shaped waveforms (due to loading errors)
    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample,2))))

    # one hot encoding of labels
    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map( lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))) )

    dataset = dataset.map(lambda sample: dict(sample, zcr=zcr(sample["waveform"])))
    dataset = dataset.map(lambda sample: (sample["waveform"], sample["one_hot_label"], sample["zcr"]))

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