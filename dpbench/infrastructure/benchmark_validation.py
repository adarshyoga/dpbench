# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Validation tools for comparing arrays."""

import logging
from numbers import Number
from typing import Union

import numpy as np


def validate(
    expected: dict[str, any],
    actual: dict[str, any],
    rel_error=1e-05,
):
    """Default validation function.

    Args:
        expected: expected values.
        actual: actual values.
        rel_error: maximum acceptable relative error.

    Returns: true, if provided data is equal.
    """
    valid = True
    for key in expected.keys():
        valid = valid and validate_two_lists_of_array(
            expected[key], actual[key], rel_error=rel_error
        )
        if not valid:
            logging.error(
                (
                    "Output did not match for {0}. "
                    + "Expected: {1} Actual: {2}"
                ).format(key, expected[key], actual[key])
            )

    return valid


def validate_two_lists_of_array(
    expected: Union[list[np.ndarray], np.ndarray],
    actual: Union[list[np.ndarray], np.ndarray],
    rel_error,
) -> bool:
    """Checks if expected equals actual with certain precision.

    Compares two arrays or two lists of arrays and validates if
    the arrays in each list have data that are either the same or close
    enough.

    Args:
        expected: list of arrays with the reference results of a specific
            benchmark
        actual: list of arrays array with the results generated by the
            framework's implementation of the benchmark
        rel_error: maximum acceptable relative error.

    Returns: true, if provided data is equal.
    """
    if not isinstance(expected, (tuple, list)):
        expected = [expected]
    if not isinstance(actual, (tuple, list)):
        actual = [actual]
    valid = True
    for r, v in zip(expected, actual):
        if not np.allclose(r, v):
            re = relative_error(r, v)
            if re < rel_error:
                continue
            valid = False
            logging.error("Relative error: {}".format(re))
            # return False
    if not valid:
        logging.error("{} did not validate!")
    return valid


def relative_error(
    ref: Union[Number, np.ndarray], val: Union[Number, np.ndarray]
) -> float:
    """Calculates relative error.

    Args:
        ref: "true" value
        val: measured value

    Returns: relative error.
    """
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        val_norm = np.linalg.norm(val)
        if val_norm == 0:
            return 0.0
        ref_norm = val_norm

    return np.linalg.norm(ref - val) / ref_norm
