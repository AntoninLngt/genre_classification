"""
    Utility functions for data processing
"""
import tensorflow as tf
import pandas as pd
import librosa

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

def audio_pipeline(filename):
    audio, _ = librosa.load(filename)
    features = []

    # Calcul du ZCR
    zcr = librosa.zero_crossings(audio)
    features.append(sum(zcr))

    # Calcul de la moyenne du Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(audio)[0]
    features.append(np.mean(spectral_centroids))
  
    # Calcul du spectral rolloff point
    rolloff = librosa.feature.spectral_rolloff(audio)[0]
    features.append(np.mean(rolloff))

    # Calcul des moyennes des MFCC
    mfcc = librosa.feature.mfcc(audio)
    for x in mfcc:
        features.append(np.mean(x))

    return features