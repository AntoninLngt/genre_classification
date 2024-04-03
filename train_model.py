"""
    Script for training demo genre classification in pure tf (no use of dzr_audio)
"""
import os
import argparse
import tensorflow as tf

from tensorflow.contrib.training import HParams

from keras_model import build_model
from data_pipeline import get_dataset


def process_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--steps_per_epoch',
                        help="Number of steps of gradient descent per epoch",
                        dest='steps_per_epoch', type=int, default=10)
    parser.add_argument('--epochs',
                        help="Number of epochs",
                        dest='epochs', type=int, default=10)

    return HParams(**parser.parse_args().__dict__)


if __name__=="__main__":
    tf.config.run_functions_eagerly(True)
    # get arguments from command line
    params = process_arguments()

    model = build_model()
    dataset = get_dataset("fma_small.csv")

    data_list = []
    for batch in dataset:
        for i in range(batch["waveform"].shape[0]):  # Assuming waveform is a tensor
            data_dict = {
                "genre": batch["genre"][i],
                "waveform": batch["waveform"][i],
                "one_hot_label": batch["one_hot_label"][i],
                "zcr": batch["zcr"][i],
                "centroid": batch["centroid"][i],
                "mfcc": batch["mfcc"][i]
            }
            data_list.append(data_dict)

    features = data_list[["waveform","one_hot_label",'zcr', 'centroid', 'mfcc']]

    labels = data_list["genre"]

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    model.fit(train_features, train_labels, steps_per_epoch=params.steps_per_epoch, epochs=params.epochs)
    # prédiction
    predictions = model.predict_classes(test_features)

    # evaluation du modèle
    _, accuracy = model.evaluate(test_features, test_labels, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))