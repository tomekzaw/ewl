import pytest
import sympy as sp
from sympy import Matrix
from sympy.physics.quantum.qubit import Qubit

from ewl import EWL
from ewl.parametrizations import U_theta_alpha_beta

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)
sin = sp.sin
cos = sp.cos

C = Matrix([[1, 0], [0, 1]])
D = Matrix([[0, i], [i, 0]])

theta1, alpha1, beta1, theta2, alpha2, beta2 = params = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2', real=True)


@pytest.fixture
def ewl() -> EWL:
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U_theta_alpha_beta(theta=pi / 2, alpha=pi / 2, beta=0)
    bob = U_theta_alpha_beta(theta=0, alpha=0, beta=0)
    return EWL(psi=psi, C=C, D=D, players=[alice, bob])


@pytest.fixture
def ewl_parametrized_00_11() -> EWL:
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U_theta_alpha_beta(theta=theta1, alpha=alpha1, beta=beta1)
    bob = U_theta_alpha_beta(theta=theta2, alpha=alpha2, beta=beta2)
    return EWL(psi=psi, C=C, D=D, players=[alice, bob])


@pytest.fixture
def ewl_parametrized_01_10() -> EWL:
    psi = (Qubit('01') + i * Qubit('10')) / sqrt2
    alice = U_theta_alpha_beta(theta=theta1, alpha=alpha1, beta=beta1)
    bob = U_theta_alpha_beta(theta=theta2, alpha=alpha2, beta=beta2)
    return EWL(psi=psi, C=C, D=D, players=[alice, bob])


def test_params_fixed(ewl: EWL):
    assert ewl.params == set()


def test_params_parametrized(ewl_parametrized_00_11: EWL):
    assert ewl_parametrized_00_11.params == set(params)


def test_fix(ewl_parametrized_00_11: EWL, ewl: EWL):
    ewl_fixed = ewl_parametrized_00_11.fix(theta1=pi / 2, alpha1=pi / 2, beta1=0, theta2=0, alpha2=0, beta2=0)
    assert ewl_fixed.psi == ewl.psi
    assert ewl_fixed.players[0] == ewl.players[0]
    assert ewl_fixed.players[1] == ewl.players[1]


def test_J(ewl: EWL):
    assert ewl.J == Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])


def test_J_H(ewl: EWL):
    assert ewl.J_H == Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, -i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, -i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])


def test_calculate_probs(ewl: EWL):
    assert ewl.probs() == Matrix([0, 0, 1 / 2, 1 / 2])


def test_amplitudes_parametrized(ewl_parametrized_00_11: EWL):
    assert ewl_parametrized_00_11.amplitudes() == Matrix([
        cos(alpha1 + alpha2) * cos(theta1 / 2) * cos(theta2 / 2) + sin(beta1 + beta2) * sin(theta1 / 2) * sin(theta2 / 2),
        sin(alpha2 - beta1) * sin(theta1 / 2) * cos(theta2 / 2) + cos(alpha1 - beta2) * cos(theta1 / 2) * sin(theta2 / 2),
        cos(alpha2 - beta1) * sin(theta1 / 2) * cos(theta2 / 2) + sin(alpha1 - beta2) * cos(theta1 / 2) * sin(theta2 / 2),
        -sin(alpha1 + alpha2) * cos(theta1 / 2) * cos(theta2 / 2) + cos(beta1 + beta2) * sin(theta1 / 2) * sin(theta2 / 2),
    ])


def test_probs_parametrized(ewl_parametrized_01_10: EWL):
    assert ewl_parametrized_01_10.probs() == Matrix([
        (cos(alpha1 - alpha2) * cos(theta1 / 2) * cos(theta2 / 2) + sin(beta1 - beta2) * sin(theta1 / 2) * sin(theta2 / 2)) ** 2,
        (sin(alpha2 + beta1) * sin(theta1 / 2) * cos(theta2 / 2) - cos(alpha1 + beta2) * cos(theta1 / 2) * sin(theta2 / 2)) ** 2,
        (cos(alpha2 + beta1) * sin(theta1 / 2) * cos(theta2 / 2) + sin(alpha1 + beta2) * cos(theta1 / 2) * sin(theta2 / 2)) ** 2,
        (-sin(alpha1 - alpha2) * cos(theta1 / 2) * cos(theta2 / 2) + cos(beta1 - beta2) * sin(theta1 / 2) * sin(theta2 / 2)) ** 2,
    ])
