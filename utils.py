"""
    Utility functions for data processing
"""
import tensorflow as tf
import pandas as pd

def load_audio_waveform(filename_tf, format="mp3", fs=44100, channel_count=2):
    """
        load waveform with tensorflow
    """
    audio_binary = tf.io.read_file(filename)
    waveform, _ = tf.audio.decode_wav(audio_binary)
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

def load_and_preprocess_audio(filename):
    """Load an audio file and apply preprocessing."""
    waveform = load_audio_waveform(filename)
    
    # Normalization
    waveform = waveform / tf.reduce_max(tf.abs(waveform))
    
    # Compute MFCCs
    mfccs = tfsignal.mfccs_from_log_mel_spectrogram(tf.math.log(tf.abs(tfsignal.stft(waveform))), 44100)
    
    # Compute spectrogram
    spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=1024, frame_step=512))
    
    return waveform, mfccs, spectrogram