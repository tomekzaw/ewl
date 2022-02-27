from typing import Dict

import pytest
import sympy as sp
from sympy import Matrix

from ewl.parametrizations import U_theta_alpha_beta, U_theta_phi_lambda, U_theta_phi

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)


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
