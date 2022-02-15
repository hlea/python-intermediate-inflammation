"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_min_string():
    """Tests for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello','there'], ['General', 'Kenobi']])

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0,0]),
        ([[1,2], [3,4], [5,6]], [3,4]),
        ([[-10,-2], [7,-9], [-3,8]], [-2,-1]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeros, positive and negative intergers"""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)),np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0,0]),
        ([[1,2], [3,4], [5,6]], [1,2]),
        ([[-10,-2], [7,-9], [-3,8]], [-10,-9]),
        ([[10.1, 2.1],[-3.3, 4.5],[5, 6.2]], [-3.3,2.1])
    ])
def test_daily_min(test, expected):
    """Test mean function works for array of zeros, positive and negative integers, positive/negative float"""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)),np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0,0]),
        ([[1,2], [3,4], [5,6]], [5,6]),
        ([[-10,-2], [7,-9], [-3,8]], [7,8]),
        ([[10.1, 2.1],[-3.3, 4.5],[5, 6.2]], [10.1,6.2])
    ])
def test_daily_max(test, expected):
    """Test mean function works for array of zeros, positive and negative integers, positive/negative float"""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)),np.array(expected))