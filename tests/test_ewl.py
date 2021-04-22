from typing import Dict

import numpy as np
import pytest
from sympy import Matrix
from sympy.physics.quantum.qubit import Qubit

from ewl import i, pi, sqrt2, number_of_qubits, sympy_to_numpy_matrix, U_theta_alpha_beta, U_theta_phi_lambda, U, J


@pytest.mark.parametrize('psi, expected', [
    ((Qubit('0') + i * Qubit('1')) / sqrt2, 1),
    ((Qubit('00') + i * Qubit('11')) / sqrt2, 2),
    ((Qubit('000') + i * Qubit('111')) / sqrt2, 3),
])
def test_number_of_qubits(psi, expected: int):
    assert number_of_qubits(psi) == expected


def test_sympy_to_numpy_matrix():
    values = [[1, 2], [3, 4]]
    assert np.allclose(sympy_to_numpy_matrix(Matrix(values)), np.array(values))


@pytest.mark.parametrize('kwargs, expected', [
    (dict(theta=0, alpha=0, beta=pi / 2), Matrix([[1, 0], [0, 1]])),
    (dict(theta=pi / 2, alpha=pi / 2, beta=0), Matrix([[i / sqrt2, i / sqrt2], [i / sqrt2, -i / sqrt2]])),
    (dict(theta=0, alpha=pi / 2), Matrix([[i, 0], [0, -i]])),
    (dict(theta=0, alpha=0, beta=0), Matrix([[1, 0], [0, 1]])),  # C
    (dict(theta=pi, alpha=0, beta=0), Matrix([[0, i], [i, 0]])),  # D
])
def test_U_theta_alpha(kwargs: Dict[str, complex], expected: Matrix):
    assert U_theta_alpha_beta(**kwargs) == expected


@pytest.mark.parametrize('kwargs, expected', [
    (dict(theta=pi, phi=0, lambda_=0), Matrix([[0, -1], [1, 0]])),
])
def test_U_theta_phi_lambda(kwargs: Dict[str, complex], expected: Matrix):
    assert U_theta_phi_lambda(**kwargs) == expected


def test_J():
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    C = U(theta=0, alpha=0, beta=0)
    D = U(theta=pi, alpha=0, beta=0)

    expected = Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])

    assert J(psi, C, D) == expected
