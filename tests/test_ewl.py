from typing import Dict

import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from sympy import Matrix
from sympy.physics.quantum.qubit import Qubit

from ewl import i, pi, sqrt2, number_of_qubits, sympy_to_numpy_matrix, \
    U_theta_alpha_beta, U_theta_phi_lambda, U_theta_phi, J, U, EWL


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
    (dict(theta=0, alpha=0, beta=0), Matrix([[1, 0], [0, 1]])),
    (dict(theta=pi, alpha=0, beta=0), Matrix([[0, i], [i, 0]])),
])
def test_U_theta_alpha(kwargs: Dict[str, complex], expected: Matrix):
    assert U_theta_alpha_beta(**kwargs) == expected


@pytest.mark.parametrize('kwargs, expected', [
    (dict(theta=0, phi=0, lambda_=0), Matrix([[1, 0], [0, 1]])),  # C
    (dict(theta=pi, phi=0, lambda_=0), Matrix([[0, -1], [1, 0]])),  # D
])
def test_U_theta_phi_lambda(kwargs: Dict[str, complex], expected: Matrix):
    assert U_theta_phi_lambda(**kwargs) == expected


@pytest.mark.parametrize('kwargs, expected', [
    (dict(theta=0, phi=pi / 2), Matrix([[i, 0], [0, -i]]))
])
def test_U_theta_phi(kwargs: Dict[str, complex], expected: Matrix):
    assert U_theta_phi(**kwargs) == expected


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


@pytest.fixture
def ewl() -> EWL:
    i = sp.I
    pi = sp.pi
    sqrt2 = sp.sqrt(2)

    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U(theta=pi / 2, alpha=pi / 2, beta=0)
    bob = U(theta=0, alpha=0, beta=0)

    return EWL(psi, [alice, bob])


def test_ewl_J(ewl: EWL):
    i = sp.I
    sqrt2 = sp.sqrt(2)

    assert ewl.J == Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])


def test_ewl_J_H(ewl: EWL):
    i = sp.I
    sqrt2 = sp.sqrt(2)

    assert ewl.J_H == Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, -i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, -i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])


def test_ewl_qc(ewl: EWL):
    i = 1j
    sqrt2 = np.sqrt(2)

    j = Operator(np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ]))

    j_h = Operator(np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, -i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, -i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ]))

    u_a = Operator(np.array([
        [i / sqrt2, i / sqrt2],
        [i / sqrt2, -i / sqrt2]
    ]))

    u_b = Operator(np.array([
        [1, 0],
        [0, 1],
    ]))

    qc = QuantumCircuit(2)
    qc.append(j, [0, 1])
    qc.barrier()
    qc.append(u_a, [0])
    qc.append(u_b, [1])
    qc.barrier()
    qc.append(j_h, [0, 1])
    qc.measure_all()

    assert ewl.qc == qc


def test_calculate_probs(ewl: EWL):
    assert ewl.probs() == Matrix([0, 0, 1 / 2, 1 / 2])
