"""
    Utility functions for data processing
"""
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd

def load_audio_waveform(filename_tf, format="mp3", fs=44100):
    """
        load waveform with tensorflow
    """
    audio_binary = tf.io.read_file(filename_tf)
    waveform, _ = tf.audio.decode_wav(audio_binary)
    # Normalization
    waveform = waveform / tf.reduce_max(tf.abs(waveform))
    return waveform

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

def mfccs(filename):
    """
    Load an audio file, compute log mel spectrogram, and then compute MFCCs.
    """
    waveform = load_audio_waveform(filename)
    # Compute spectrogram
    spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=1024, frame_step=512))
    # Compute log mel spectrogram
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40, 
        num_spectrogram_bins=tf.shape(spectrogram)[-1], 
        sample_rate=44100, 
        lower_edge_hertz=20.0, 
        upper_edge_hertz=8000.0
    )
    mel_spectrogram = tf.transpose(mel_spectrogram)
    range_tensor = tf.cast(tf.range(1, tf.shape(spectrogram)[-1] + 1), tf.float32)
    range_tensor = tf.reshape(range_tensor, [1, -1])
    mel_spectrogram *= range_tensor
    mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    # Compute MFCCs from log mel spectrogram using TensorFlow signal module
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    return mfccs

def spectrogram(filename):
    """
    Compute spectrogram from the audio file.
    """
    waveform = load_audio_waveform(filename)
    spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=1024, frame_step=512))
    return spectrogram