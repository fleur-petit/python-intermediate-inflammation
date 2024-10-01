"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 4], [4, 1, 9] ], [3, 3, 6]),
        ([ [4, -2, 5], [6, -6, 2], [-4, -1, 11] ], [2, -3, 6]),
    ])
def test_daily_mean(test, expected):
    """Test that mean function works for an array of positive integers, zeros and negative numbers."""
    from inflammation.models import daily_mean
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 4], [4, 1, 9] ], [4, 6, 9]),
        ([ [4, -2, 5], [6, -6, 2], [-4, -1, 11] ], [6, -1, 11]),
    ])
def test_daily_max(test, expected):
    """Test that max function works for an array of positive integers, zeros and negative numbers."""
    from inflammation.models import daily_max

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 4], [4, 1, 9] ], [1, 1, 4]),
        ([ [4, -2, 5], [6, -6, 2], [-4, -1, 11] ], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test that min function works for an array of positive integers, zeros and negative."""
    from inflammation.models import daily_min

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

