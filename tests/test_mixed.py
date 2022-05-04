from typing import Any

import pytest
import sympy as sp
from sympy.physics.quantum.qubit import Qubit

from ewl.mixed import MixedEWL, MixedStrategy
from ewl.parametrizations import U_theta_phi_alpha

i = sp.I
pi = sp.pi

psi = (Qubit('00') + i * Qubit('11')) / sp.sqrt(2)

C = sp.Matrix([[1, 0], [0, 1]])
D = sp.Matrix([[0, i], [i, 0]])

payoff_matrix = sp.Array([
    [
        [3, 0],
        [5, 1],
    ],
    [
        [3, 5],
        [0, 1],
    ],
])

theta_A, phi_A, alpha_A = sp.symbols('theta_A phi_A alpha_A', real=True)

A_hat = U_theta_phi_alpha(theta=theta_A, phi=phi_A, alpha=alpha_A)
B_hat = U_theta_phi_alpha(theta=theta_A + pi, phi=alpha_A, alpha=phi_A - pi / 2)
A_hat_prim = U_theta_phi_alpha(theta=theta_A, phi=phi_A - pi / 2, alpha=alpha_A - pi / 2)
B_hat_prim = U_theta_phi_alpha(theta=theta_A + pi, phi=alpha_A - pi / 2, alpha=phi_A - pi)

gamma_A, gamma_B = sp.symbols('gamma_A gamma_B', real=True)

Expr = Any


@pytest.fixture
def alice() -> MixedStrategy:
    return MixedStrategy([
        (sp.cos(gamma_A / 2) ** 2, A_hat),
        (sp.sin(gamma_A / 2) ** 2, A_hat_prim),
    ])


@pytest.fixture
def bob() -> MixedStrategy:
    return MixedStrategy([
        (sp.cos(gamma_B / 2) ** 2, B_hat),
        (sp.sin(gamma_B / 2) ** 2, B_hat_prim),
    ])


@pytest.fixture
def mixed_ewl(alice: MixedStrategy, bob: MixedStrategy) -> MixedEWL:
    return MixedEWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)


@pytest.fixture
def mixed_ewl_fixed(mixed_ewl: MixedEWL) -> MixedEWL:
    return mixed_ewl.fix(theta_A=pi / 2, phi_A=0, alpha_A=0)


def test_kraus() -> None:
    # TODO: implement
    pass


def test_MixedStrategy_invalid_sum() -> None:
    with pytest.raises(AssertionError):
        MixedStrategy([(0.9, C), (0.2, D)], check_sum=True)


def test_MixedStrategy_params(alice: MixedStrategy, bob: MixedStrategy) -> None:
    assert alice.params == {gamma_A, theta_A, alpha_A, phi_A}
    assert bob.params == {gamma_B, theta_A, alpha_A, phi_A}


def test_MixedStrategy_fix(alice: MixedStrategy) -> None:
    # TODO: implement
    pass


def test_MixedEWL_number_of_players(mixed_ewl: MixedEWL) -> None:
    assert mixed_ewl.number_of_players == 2


def test_MixedEWL_params(mixed_ewl: MixedEWL, mixed_ewl_fixed: MixedEWL) -> None:
    assert mixed_ewl.params == {gamma_A, gamma_B, theta_A, alpha_A, phi_A}
    assert mixed_ewl_fixed.params == {gamma_A, gamma_B}


def test_MixedEWL_fix(mixed_ewl: MixedEWL) -> None:
    # TODO: implement
    pass


def test_MixedEWL_amplitudes(mixed_ewl: MixedEWL) -> None:
    with pytest.raises(NotImplementedError):
        mixed_ewl.amplitudes()


def test_MixedEWL_density_matrix(mixed_ewl: MixedEWL) -> None:
    # TODO: implement
    pass


def test_MixedEWL_probs(mixed_ewl: MixedEWL) -> None:
    # TODO: implement
    pass


def test_MixedEWL_probs_sum(mixed_ewl_fixed: MixedEWL) -> None:
    assert sum(mixed_ewl_fixed.probs()) == 1


def test_MixedEWL_payoff_function(mixed_ewl: MixedEWL) -> None:
    # TODO: implement
    pass
