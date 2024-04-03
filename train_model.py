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

    # get arguments from command line
    params = process_arguments()
    model = build_model()
    dataset = get_dataset("fma_small.csv")
    model.fit(dataset, steps_per_epoch=params.steps_per_epoch, epochs=params.epochs)