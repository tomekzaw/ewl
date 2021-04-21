from math import sqrt
from typing import Dict

import numpy as np
import pytest

from ewl import i, pi, sqrt2, ket, is_unit_vector, number_of_qbits, U_theta_alpha_beta, U_theta_phi_lambda, U, J


@pytest.mark.parametrize('base_state, expected', [
    ('0', [1, 0]),
    ('1', [0, 1]),
    ('00', [1, 0, 0, 0]),
    ('01', [0, 1, 0, 0]),
    ('10', [0, 0, 1, 0]),
    ('11', [0, 0, 0, 1]),
    ('000', [1] + [0] * 7),
    ('011', [0] * 3 + [1] + [0] * 4),
    ('111', [0] * 7 + [1]),
    ('01010', [0] * 10 + [1] + [0] * 21),
])
def test_ket(base_state: str, expected: np.array):
    assert np.array_equal(ket(base_state), np.array(expected))


@pytest.mark.parametrize('invalid_state', ['', '0a'])
def test_ket_error(invalid_state: str):
    with pytest.raises(AssertionError):
        ket(invalid_state)


@pytest.mark.parametrize('vec, expected', [
    (ket('10'), True),
    (ket('10') * 2, False),
    ((ket('00') + i * ket('11')) / sqrt2, True),
    ((ket('00') + i * ket('11')) / sqrt(3), False),
    ((ket('000') + i * ket('111')) / sqrt2, True),
])
def test_is_unit_vector(vec: np.array, expected: bool):
    assert is_unit_vector(vec) == expected


@pytest.mark.parametrize('psi, expected', [
    (ket('1'), 1),
    (ket('10'), 2),
    (ket('101'), 3),
    (ket('1011'), 4),
    (ket('10110'), 5),
])
def test_number_of_qbits(psi: np.array, expected: int):
    assert number_of_qbits(psi) == expected


@pytest.mark.parametrize('kwargs, expected', [
    (dict(theta=0, alpha=0, beta=pi / 2), np.array([[1, 0], [0, 1]])),
    (dict(theta=0, alpha=pi / 2), np.array([[i, 0], [0, -i]])),
    (dict(theta=0, alpha=0, beta=0), np.array([[1, 0], [0, 1]])),  # C
    (dict(theta=pi, alpha=0, beta=0), np.array([[0, i], [i, 0]])),  # D
])
def test_U_theta_alpha(kwargs: Dict[str, complex], expected: np.array):
    assert np.allclose(U_theta_alpha_beta(**kwargs), expected)


@pytest.mark.parametrize('kwargs, expected', [
    (dict(theta=pi, phi=0, lambda_=0), np.array([[0, -1], [1, 0]])),
])
def test_U_theta_phi_lambda(kwargs: Dict[str, complex], expected: np.array):
    assert np.allclose(U_theta_phi_lambda(**kwargs), expected)


def test_J():
    assert np.allclose(J(
        psi=np.array([1, 0, 0, i]) / sqrt2,
        C=U(theta=0, alpha=0, beta=0),
        D=U(theta=pi, alpha=0, beta=0)
    ), np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ]))
