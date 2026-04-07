# -*- coding: utf-8 -*-
"""Test utility functions: n_saved_timesteps, result_len, timestep_boundaries, discontify."""

import numpy as np
import pytest

from pydgq.solver.types import DTYPE
from pydgq.solver.odesolve import n_saved_timesteps, result_len, timestep_boundaries
from pydgq.utils.discontify import discontify


# ---------------------------------------------------------------------------
# n_saved_timesteps
# ---------------------------------------------------------------------------

def test_n_saved_timesteps_with_ic():
    # save_from=0: IC + all 100 timesteps
    assert n_saved_timesteps(100, 0) == 101


def test_n_saved_timesteps_from_first():
    # save_from=1: all 100 timesteps, no IC
    assert n_saved_timesteps(100, 1) == 100


def test_n_saved_timesteps_from_midpoint():
    # save_from=50: timesteps 50..100 => 51
    assert n_saved_timesteps(100, 50) == 51


# ---------------------------------------------------------------------------
# result_len
# ---------------------------------------------------------------------------

def test_result_len_with_ic():
    # 1 (IC) + 100*1 = 101
    assert result_len(100, 0, 1) == 101


def test_result_len_no_ic():
    # (100 - 0)*1 = 100
    assert result_len(100, 1, 1) == 100


def test_result_len_interp():
    # 1 (IC) + 100*2 = 201
    assert result_len(100, 0, 2) == 201


def test_result_len_validation_nt():
    with pytest.raises(ValueError):
        result_len(0, 0)


def test_result_len_validation_save_from_negative():
    with pytest.raises(ValueError):
        result_len(10, -1)


def test_result_len_validation_save_from_too_large():
    with pytest.raises(ValueError):
        result_len(10, 11)


def test_result_len_validation_interp_zero():
    with pytest.raises(ValueError):
        result_len(10, 0, 0)


# ---------------------------------------------------------------------------
# timestep_boundaries
# ---------------------------------------------------------------------------

def test_timestep_boundaries_basic():
    startj, endj = timestep_boundaries(10, 1, 1)
    assert len(startj) == 10
    assert startj[0] == 0
    assert endj[0] == 1


def test_timestep_boundaries_with_ic():
    startj, endj = timestep_boundaries(10, 0, 1)
    # IC slot + 10 timesteps = 11 entries
    assert len(startj) == 11
    # IC is always a single point
    assert startj[0] == 0
    assert endj[0] == 1
    # First actual timestep
    assert startj[1] == 1
    assert endj[1] == 2


def test_timestep_boundaries_interp():
    startj, endj = timestep_boundaries(5, 1, 3)
    assert len(startj) == 5
    # Each timestep occupies 3 slots
    assert startj[0] == 0
    assert endj[0] == 3
    assert startj[1] == 3
    assert endj[1] == 6


# ---------------------------------------------------------------------------
# discontify
# ---------------------------------------------------------------------------

def test_discontify_nan():
    data = np.arange(5, dtype=DTYPE)
    idxs = np.array([1, 3], dtype=np.intc)
    result = discontify(data, idxs, fill="nan")
    # 5 original + 2 inserted = 7
    assert len(result) == 7
    # NaN inserted after index 1 (at position 2 in output)
    assert np.isnan(result[2])
    # NaN inserted after index 3 (at position 5 in output)
    assert np.isnan(result[5])


def test_discontify_prev():
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=DTYPE)
    idxs = np.array([1], dtype=np.intc)
    result = discontify(data, idxs, fill="prev")
    assert len(result) == 5
    # Inserted value copies the element at position 1
    assert result[2] == 20.0


def test_discontify_empty_idxs():
    data = np.arange(5, dtype=DTYPE)
    idxs = np.array([], dtype=np.intc)
    result = discontify(data, idxs, fill="nan")
    np.testing.assert_array_equal(result, data)


def test_discontify_invalid_fill():
    data = np.arange(5, dtype=DTYPE)
    idxs = np.array([1], dtype=np.intc)
    with pytest.raises(ValueError, match="Unknown fill"):
        discontify(data, idxs, fill="invalid")
