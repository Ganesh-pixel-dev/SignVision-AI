"""Loading the landmark CSV and splitting it without leaking near-duplicates."""

import itertools
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "training", "dataset", "landmarks.csv")

EMBARGO = 10  # samples dropped between the train and test part of one recording
TEST_FRACTION = 0.2


def load(path=DATA_CSV):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, df["label"].values


def recording_blocks(labels):
    """Group id per row: a run of consecutive rows with one label is one recording.

    The collector appends rows in time order, so a run is one continuous
    hold of one sign. The CSV has no session column; this is the closest
    thing to one.
    """
    ids = []
    for block, (_, run) in enumerate(itertools.groupby(labels)):
        ids.extend([block] * len(list(run)))
    return np.array(ids)


def temporal_split(labels, test_fraction=TEST_FRACTION, embargo=EMBARGO):
    """Hold out the last part of every recording block.

    Rows inside a block are near-duplicates of their neighbours, so a
    random split lets the test set see almost the same sample as the
    train set. Taking the tail of each block, with a gap, is a fairer
    test but still the same session, same hand, same lighting.
    """
    blocks = recording_blocks(labels)
    train, test = [], []
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        n_test = int(len(idx) * test_fraction)
        if n_test < 5:
            train.extend(idx)
            continue
        test.extend(idx[-n_test:])
        train.extend(idx[: len(idx) - n_test - embargo])
    return np.array(train), np.array(test)
