from typing import Dict

import pytest
import sympy as sp
from sympy import Matrix

from ewl.parametrizations import U_theta_alpha_beta, U_theta_phi_lambda, U_theta_phi, U_Frackiewicz_Pykacz, U

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)


def test_U_theta_alpha_beta():
    theta, alpha, beta = sp.symbols('theta alpha beta')
    actual = U_theta_alpha_beta(theta=theta, alpha=alpha, beta=beta)
    expected = Matrix([
        [sp.exp(i * alpha) * sp.cos(theta / 2), i * sp.exp(i * beta) * sp.sin(theta / 2)],
        [i * sp.exp(-i * beta) * sp.sin(theta / 2), sp.exp(-i * alpha) * sp.cos(theta / 2)]
    ])
    assert actual == expected


def test_U_theta_phi_lambda():
    theta, phi, lambda_ = sp.symbols('theta phi lambda')
    actual = U_theta_phi_lambda(theta=theta, phi=phi, lambda_=lambda_)
    expected = Matrix([
        [sp.exp(-i * (phi + lambda_) / 2) * sp.cos(theta / 2), -sp.exp(-i * (phi - lambda_) / 2) * sp.sin(theta / 2)],
        [sp.exp(i * (phi - lambda_) / 2) * sp.sin(theta / 2), sp.exp(i * (phi + lambda_) / 2) * sp.cos(theta / 2)]
    ])
    assert actual == expected


def test_U_theta_phi():
    theta, phi = sp.symbols('theta phi')
    actual = U_theta_phi(theta=theta, phi=phi)
    expected = Matrix([
        [sp.exp(i * phi) * sp.cos(theta / 2), sp.sin(theta / 2)],
        [-sp.sin(theta / 2), sp.exp(-i * phi) * sp.cos(theta / 2)]
    ])
    assert actual == expected


def test_U_Frackiewicz_Pykacz():
    theta, phi = sp.symbols('theta phi')
    actual = U_Frackiewicz_Pykacz(theta=theta, phi=phi)
    expected = Matrix([
        [sp.exp(i * phi) * sp.cos(theta / 2), i * sp.exp(i * phi) * sp.sin(theta / 2)],
        [i * sp.exp(-i * phi) * sp.sin(theta / 2), sp.exp(-i * phi) * sp.cos(theta / 2)]
    ])
    assert actual == expected


@pytest.mark.parametrize('kwargs, expected', [
    # theta, alpha, beta
    (dict(theta=0, alpha=0, beta=pi / 2), Matrix([[1, 0], [0, 1]])),
    (dict(theta=pi / 2, alpha=pi / 2, beta=0), Matrix([[i / sqrt2, i / sqrt2], [i / sqrt2, -i / sqrt2]])),
    (dict(theta=0, alpha=pi / 2), Matrix([[i, 0], [0, -i]])),
    (dict(theta=0, alpha=0, beta=0), Matrix([[1, 0], [0, 1]])),
    (dict(theta=pi, alpha=0, beta=0), Matrix([[0, i], [i, 0]])),

    # theta, phi, lambda
    (dict(theta=0, phi=0, lambda_=0), Matrix([[1, 0], [0, 1]])),  # C
    (dict(theta=pi, phi=0, lambda_=0), Matrix([[0, -1], [1, 0]])),  # D

    # theta, phi
    (dict(theta=0, phi=pi / 2), Matrix([[i, 0], [0, -i]]))
])
def test_U(kwargs: Dict[str, complex], expected: Matrix):
    assert U(**kwargs) == expected
