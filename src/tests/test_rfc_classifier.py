import numpy as np
import pandas as pd
from unittest.mock import patch
from dataprocessing_of_gui import rfc


class DummyModel:
    def predict(self, X):
        return np.ones(len(X))


@patch("dataprocessing_of_gui.load_selected_model")
def test_rfc_classifier_basic(mock_load_model):
    mock_load_model.return_value = DummyModel()

    X = pd.DataFrame(np.random.rand(5, 3))
    y = pd.Series([1, 1, 1, 1, 1])

    accuracy, precision, recall, f1, pane = rfc(X, y)

    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    assert pane is not None
