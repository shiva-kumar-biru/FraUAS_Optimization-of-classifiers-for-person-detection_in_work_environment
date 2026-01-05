import numpy as np
from dataprocessing_of_gui import autocorrelation


def test_autocorrelation_output_length():
    x = np.array([1, 2, 3, 4])

    result = autocorrelation(x)

    assert result is not None
    assert len(result) == len(x)


def test_autocorrelation_first_value():
    x = np.array([1, 2, 3])

    result = autocorrelation(x)

    expected = 1**2 + 2**2 + 3**2
    assert result[0] == expected
