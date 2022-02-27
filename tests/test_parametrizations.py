from typing import Dict

import pytest
import sympy as sp
from sympy import Matrix

from ewl.parametrizations import U_theta_alpha_beta, U_theta_phi_lambda, U_theta_phi_alpha, \
    U_Eisert_Wilkens_Lewenstein, U_Frackiewicz_Pykacz, U

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)
sin = sp.sin
cos = sp.cos
exp = sp.exp


def test_U_theta_alpha_beta():
    theta, alpha, beta = sp.symbols('theta alpha beta')
    actual = U_theta_alpha_beta(theta=theta, alpha=alpha, beta=beta)
    expected = Matrix([
        [exp(i * alpha) * cos(theta / 2), i * exp(i * beta) * sin(theta / 2)],
        [i * exp(-i * beta) * sin(theta / 2), exp(-i * alpha) * cos(theta / 2)]
    ])
    assert actual == expected


def test_U_theta_phi_lambda():
    theta, phi, lambda_ = sp.symbols('theta phi lambda')
    actual = U_theta_phi_lambda(theta=theta, phi=phi, lambda_=lambda_)
    expected = Matrix([
        [exp(-i * (phi + lambda_) / 2) * cos(theta / 2), -exp(-i * (phi - lambda_) / 2) * sin(theta / 2)],
        [exp(i * (phi - lambda_) / 2) * sin(theta / 2), exp(i * (phi + lambda_) / 2) * cos(theta / 2)]
    ])
    assert actual == expected


def test_U_theta_phi_alpha():
    theta, phi, alpha = sp.symbols('theta phi alpha')
    actual = U_theta_phi_alpha(theta=theta, phi=phi, alpha=alpha)
    expected = Matrix([
        [exp(-i * phi) * cos(theta / 2), exp(i * alpha) * sin(theta / 2)],
        [-exp(-i * alpha) * sin(theta / 2), exp(i * phi) * cos(theta / 2)]
    ])
    assert actual == expected


def test_U_Eisert_Wilkens_Lewenstein():
    theta, phi = sp.symbols('theta phi')
    actual = U_Eisert_Wilkens_Lewenstein(theta=theta, phi=phi)
    expected = Matrix([
        [exp(i * phi) * cos(theta / 2), sin(theta / 2)],
        [-sin(theta / 2), exp(-i * phi) * cos(theta / 2)]
    ])
    assert actual == expected


def test_U_Frackiewicz_Pykacz():
    theta, phi = sp.symbols('theta phi')
    actual = U_Frackiewicz_Pykacz(theta=theta, phi=phi)
    expected = Matrix([
        [exp(i * phi) * cos(theta / 2), i * exp(i * phi) * sin(theta / 2)],
        [i * exp(-i * phi) * sin(theta / 2), exp(-i * phi) * cos(theta / 2)]
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
