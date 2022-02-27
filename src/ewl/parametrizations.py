import sympy as sp
from sympy import Matrix

i = sp.I
pi = sp.pi


def U_theta_alpha_beta(*, theta, alpha, beta=3 * pi / 2) -> Matrix:
    return Matrix([
        [sp.exp(i * alpha) * sp.cos(theta / 2), i * sp.exp(i * beta) * sp.sin(theta / 2)],
        [i * sp.exp(-i * beta) * sp.sin(theta / 2), sp.exp(-i * alpha) * sp.cos(theta / 2)]
    ])


def U_theta_phi_lambda(*, theta, phi, lambda_) -> Matrix:
    return Matrix([
        [sp.exp(-i * (phi + lambda_) / 2) * sp.cos(theta / 2), -sp.exp(-i * (phi - lambda_) / 2) * sp.sin(theta / 2)],
        [sp.exp(i * (phi - lambda_) / 2) * sp.sin(theta / 2), sp.exp(i * (phi + lambda_) / 2) * sp.cos(theta / 2)]
    ])


def U_theta_phi(*, theta, phi) -> Matrix:
    # original parametrization from "Quantum Games and Quantum Strategies" by J. Eisert, M. Wilkens, M. Lewenstein
    return Matrix([
        [sp.exp(i * phi) * sp.cos(theta / 2), sp.sin(theta / 2)],
        [-sp.sin(theta / 2), sp.exp(-i * phi) * sp.cos(theta / 2)]
    ])


def U_Frackiewicz_Pykacz(theta, phi) -> Matrix:
    """
    Full SU(2) parametrization from "Quantum Games with Strategies Induced by Basis Change Rules"
    by Piotr Frąckiewicz and Jarosław Pykacz (DOI:10.1007/s10773-017-3423-6).
    https://link.springer.com/content/pdf/10.1007%2Fs10773-017-3423-6.pdf
    :param theta: [0, PI]
    :param phi: [0, 2*PI]
    :return:
    """
    return Matrix([
        [sp.exp(i * phi) * sp.cos(theta / 2), i * sp.exp(i * phi) * sp.sin(theta / 2)],
        [i * sp.exp(-i * phi) * sp.sin(theta / 2), sp.exp(-i * phi) * sp.cos(theta / 2)]
    ])


def U(*args, **kwargs) -> Matrix:
    if args:
        raise Exception('Please use keyword arguments')
    if set(kwargs) in [{'theta', 'alpha'}, {'theta', 'alpha', 'beta'}]:
        return U_theta_alpha_beta(**kwargs)
    if set(kwargs) == {'theta', 'phi', 'lambda_'}:
        return U_theta_phi_lambda(**kwargs)
    if set(kwargs) == {'theta', 'phi'}:
        return U_theta_phi(**kwargs)
    raise Exception('Invalid parametrization')
