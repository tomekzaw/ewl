import numpy as np
import pytest
import sympy as sp
from sympy import Matrix, Symbol
from sympy.physics.quantum.qubit import Qubit

from ewl.utils import number_of_qubits, amplitude_to_prob, sympy_to_numpy_matrix

i = sp.I
sqrt2 = sp.sqrt(2)
sin = sp.sin
cos = sp.cos
exp = sp.exp


@pytest.mark.parametrize('psi, expected', [
    ((Qubit('0') + i * Qubit('1')) / sqrt2, 1),
    ((Qubit('00') + i * Qubit('11')) / sqrt2, 2),
    ((Qubit('000') + i * Qubit('111')) / sqrt2, 3),
])
def test_number_of_qubits(psi, expected: int):
    assert number_of_qubits(psi) == expected


def test_amplitude_to_prob_real():
    A = Symbol('A', real=True)
    assert amplitude_to_prob(A) == A ** 2


def test_sympy_to_numpy_matrix():
    values = [[1, 2], [3, 4]]
    actual = sympy_to_numpy_matrix(Matrix(values))
    expected = np.array(values)
    assert np.allclose(actual, expected)
