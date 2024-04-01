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

def get_dataset(input_csv, batch_size=8):

    # build dataset from csv file
    dataset = dataset_from_csv(input_csv)

    # add directory in the filename
    dataset = dataset.map(lambda sample: dict(sample, filename=tf.string_join([DATASET_DIR, sample["filename"]])))

    n_sample = 11025
    # load audio and take first quarter of second only
    dataset = dataset.map(lambda sample: dict(sample, waveform=load_audio_waveform(sample["filename"])[:n_sample,:]), num_parallel_calls=32)

    # Filter out badly shaped waveforms (due to loading errors)
    dataset = dataset.filter(lambda sample: tf.reduce_all(tf.equal(tf.shape(sample["waveform"]), (n_sample,2))))

    # one hot encoding of labels
    label_list = ["Electronic", "Folk", "Hip-Hop", "Indie-Rock", "Jazz", "Old-Time", "Pop", "Psych-Rock", "Punk", "Rock"]
    dataset = dataset.map(lambda sample: dict(sample, one_hot_label=one_hot_label(sample["genre"], tf.constant(label_list))))

    # Make batch
    dataset = dataset.batch(batch_size)

    return dataset


# test dataset data generation
if __name__=="__main__":

    dataset = get_dataset("fma_small.csv")
    batch = dataset.make_one_shot_iterator().get_next()

    mel_specs = []

    for l in label_list:

        for audio in dataset[l]:

            y = audio[0]
            sr = audio[1]

            spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
            spect = librosa.power_to_db(spect, ref=np.max)
        
        # On modifie la taille des images 128 x 660 en gardant les paramètres proposés dans l'article initial
            if spect.shape[1] != 660:
                spect.resize(128,660, refcheck=False)

            mel_specs.append(spect)
        
    X = np.array(mel_specs)
    y_cnn = []

    for i in range(len(genres)):
        y_cnn += 100*[i] # On a 100 images pour chaque genre

    y_cnn = np.array(y_cnn)

    x_cnn_train /= -80
    x_cnn_test /= -80

    x_cnn_train = x_cnn_train.reshape(x_cnn_train.shape[0], 128, 660, 1)
    x_cnn_test = x_cnn_test.reshape(x_cnn_test.shape[0], 128, 660, 1)

    

    with tf.Session() as sess:

        # Evaluate first batch
        batch_value = sess.run(batch)
        print("Training dataset generated a batch with:")
        for el in batch_value:
            print(x_cnn_train.shape)
            print(y_cnn_train.shape)
            print(f"A {type(el)} with shape {el.shape}.")
            
