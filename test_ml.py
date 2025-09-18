import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from ml.model import load_model, compute_model_metrics

import pickle

def test_test_train_split(data):
    """
    # Tests to see if the test/train split have the correct number of rows
    """
    test_size=0.25
    X_train, X_test = train_test_split(data, test_size=test_size)

    X_test_check = np.ceil(data.shape[0] * test_size)
    X_train_check = data.shape[0] - X_test_check

    # Only X_train needs to be checked as we use X_test_check to derive X_train_check
    assert X_train_check == X_train.shape[0]
    pass


def test_valid_model():
    """
    # Ensures the saved model is a RandomForestClassifier model
    """
    model_path = os.path.join("", "model", "model.pkl")
    model = load_model(
        model_path
    )
    assert type(model) == RandomForestClassifier
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # Tests the compute_model_metrics function to ensure it calculates the metrics properly
    """
    y = [1, 0, 1, 1, 0, 1]
    y_preds = [1, 0, 0, 1, 0, 0]

    p, r, fb = compute_model_metrics(y, y_preds)
    assert (p == 1.0, r == 0.5, fb == 1/3)
    pass

