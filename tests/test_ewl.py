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
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U(theta=pi / 2, alpha=pi / 2, beta=0)
    bob = U(theta=0, alpha=0, beta=0)
    return EWL(psi, [alice, bob])


@pytest.fixture
def ewl_parametrized() -> EWL:
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    theta1, alpha1, beta1, theta2, alpha2, beta2 = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2')
    alice = U(theta=theta1, alpha=alpha1, beta=beta1)
    bob = U(theta=theta2, alpha=alpha2, beta=beta2)
    return EWL(psi, [alice, bob])


@pytest.fixture
def ewl_parametrized_01_10() -> EWL:
    psi = (Qubit('01') + i * Qubit('10')) / sqrt2
    theta1, alpha1, beta1, theta2, alpha2, beta2 = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2')
    alice = U(theta=theta1, alpha=alpha1, beta=beta1)
    bob = U(theta=theta2, alpha=alpha2, beta=beta2)
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


def test_ewl_parametrized_qc(ewl_parametrized: EWL):
    with pytest.raises(Exception):
        ewl_parametrized.qc


def test_calculate_probs(ewl: EWL):
    assert ewl.probs() == Matrix([0, 0, 1 / 2, 1 / 2])


def test_amplitudes_parametrized(ewl_parametrized: EWL):
    theta1, alpha1, beta1, theta2, alpha2, beta2 = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2')
    sin, cos = sp.sin, sp.cos

    amplitudes = ewl_parametrized.amplitudes(simplify=True)

    assert amplitudes[0] == \
           cos(alpha1 + alpha2) * cos(theta1 / 2) * cos(theta2 / 2) + \
           sin(beta1 + beta2) * sin(theta1 / 2) * sin(theta2 / 2)

    assert amplitudes[1] == \
           sin(alpha2 - beta1) * sin(theta1 / 2) * cos(theta2 / 2) + \
           cos(alpha1 - beta2) * cos(theta1 / 2) * sin(theta2 / 2)

    assert amplitudes[2] == \
           cos(alpha2 - beta1) * sin(theta1 / 2) * cos(theta2 / 2) + \
           sin(alpha1 - beta2) * cos(theta1 / 2) * sin(theta2 / 2)

    assert amplitudes[3] == \
           -sin(alpha1 + alpha2) * cos(theta1 / 2) * cos(theta2 / 2) + \
           cos(beta1 + beta2) * sin(theta1 / 2) * sin(theta2 / 2)


def test_probs_parametrized(ewl_parametrized_01_10: EWL):
    theta1, alpha1, beta1, theta2, alpha2, beta2 = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2')
    sin, cos, abs = sp.sin, sp.cos, sp.Abs

    probs = ewl_parametrized_01_10.probs(simplify=True)

    assert probs[0] == abs(cos(alpha1 - alpha2) * cos(theta1 / 2) * cos(theta2 / 2) +  # noqa: W504
                           sin(beta1 - beta2) * sin(theta1 / 2) * sin(theta2 / 2)) ** 2

    assert probs[1] == abs(-sin(alpha2 + beta1) * sin(theta1 / 2) * cos(theta2 / 2) +  # noqa: W504
                           cos(alpha1 + beta2) * cos(theta1 / 2) * sin(theta2 / 2)) ** 2

    assert probs[2] == abs(cos(alpha2 + beta1) * sin(theta1 / 2) * cos(theta2 / 2) +  # noqa: W504
                           sin(alpha1 + beta2) * cos(theta1 / 2) * sin(theta2 / 2)) ** 2

    assert probs[3] == abs(-sin(alpha1 - alpha2) * cos(theta1 / 2) * cos(theta2 / 2) +  # noqa: W504
                           cos(beta1 - beta2) * sin(theta1 / 2) * sin(theta2 / 2)) ** 2


def test_params_fixed(ewl: EWL):
    assert ewl.params == set()


def test_params_parametrized(ewl_parametrized: EWL):
    assert ewl_parametrized.params == set(sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2'))


def test_fix(ewl: EWL, ewl_parametrized: EWL):
    ewl_fixed = ewl_parametrized.fix(theta1=pi / 2, alpha1=pi / 2, beta1=0,
                                     theta2=0, alpha2=0, beta2=0)

    assert ewl_fixed.J == ewl.J
    assert ewl_fixed.strategies[0] == ewl.strategies[0]
    assert ewl_fixed.strategies[1] == ewl.strategies[1]
