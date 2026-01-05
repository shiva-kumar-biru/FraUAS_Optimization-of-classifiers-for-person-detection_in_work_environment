import io

import numpy as np
import pandas as pd

from dataprocessing_of_gui import file_processing


def create_dummy_csv():
    data = np.random.rand(1, 20)
    df = pd.DataFrame(data)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    return buffer


def test_file_processing_returns_data():
    csv_file = create_dummy_csv()

    normalized_data, labels = file_processing(csv_file)

    assert normalized_data is not None
    assert labels is not None


def test_file_processing_normalization_range():
    csv_file = create_dummy_csv()

    normalized_data, _ = file_processing(csv_file)

    # DataFrame should exist
    assert normalized_data is not None

    # If data is not empty, then it must be normalized
    if not normalized_data.empty:
        assert normalized_data.min().min() >= 0
        assert normalized_data.max().max() <= 1
