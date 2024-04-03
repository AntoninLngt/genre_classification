"""
    Utility functions for data processing
"""
import tensorflow as tf
import librosa
import pandas as pd


def load_audio_waveform(filename_tf, format="mp3", fs=44100, channel_count=2):
    """
        load waveform with tensorflow
    """
    audio_binary = tf.read_file(filename_tf)
    return tf.contrib.ffmpeg.decode_audio(audio_binary, file_format=format, samples_per_second=fs, channel_count=channel_count)


def one_hot_label(label_string_tf, label_list_tf, dtype=tf.float32):
    """
        Transform string label to one hot vector.
    """
    return tf.cast(tf.equal(label_list_tf, label_string_tf), dtype)

def dataset_from_csv(csv_path, **kwargs):
    """
        Load dataset from a csv file.
        kwargs are forwarded to the pandas.read_csv function.
    """
    df = pd.read_csv(csv_path, **kwargs)

    dataset = tf.data.Dataset.from_tensor_slices({key:df[key].values for key in df })
    return dataset


def zcr(filename):
    audio = load_audio_waveform(filename)
    zcr = librosa.zero_crossings(audio)
    return zcr

def spectral_centroids(filename):
    audio = load_audio_waveform(filename)
    spectral_centroids = librosa.feature.spectral_centroid(audio)[0]
    return spectral_centroids

def mfcc(filename):
    audio = load_audio_waveform(filename)
    mfcc = librosa.feature.mfcc(audio)
    return mfccs

def rolloff(filename):
    audio = load_audio_waveform(filename)
    rolloff = librosa.feature.spectral_rolloff(audio)
    return rolloff