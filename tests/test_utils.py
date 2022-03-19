import numpy as np
import pytest
import sympy as sp
from sympy import Matrix, Symbol
from sympy.physics.quantum import TensorProduct
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


def test_tensor_product():
    a11, a12, a21, a22, b11, b12, b21, b22 = sp.symbols('a11 a12 a21 a22 b11 b12 b21 b22')
    A = sp.Matrix([
        [a11, a12],
        [a21, a22],
    ])
    B = sp.Matrix([
        [b11, b12],
        [b21, b22],
    ])
    actual = TensorProduct(A, B)
    expected = sp.Matrix([
        [a11 * b11, a11 * b12, a12 * b11, a12 * b12],
        [a11 * b21, a11 * b22, a12 * b21, a12 * b22],
        [a21 * b11, a21 * b12, a22 * b11, a22 * b12],
        [a21 * b21, a21 * b22, a22 * b21, a22 * b22],
    ])
    assert actual == expected
