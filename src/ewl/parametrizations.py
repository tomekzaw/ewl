import sympy as sp
from sympy import Matrix

i = sp.I
pi = sp.pi
sin = sp.sin
cos = sp.cos
exp = sp.exp


def U_theta_alpha_beta(*, theta, alpha, beta=3 * pi / 2) -> Matrix:
    return Matrix([
        [exp(i * alpha) * cos(theta / 2), i * exp(i * beta) * sin(theta / 2)],
        [i * exp(-i * beta) * sin(theta / 2), exp(-i * alpha) * cos(theta / 2)]
    ])


def U_theta_phi_lambda(*, theta, phi, lambda_) -> Matrix:
    return Matrix([
        [exp(-i * (phi + lambda_) / 2) * cos(theta / 2), -exp(-i * (phi - lambda_) / 2) * sin(theta / 2)],
        [exp(i * (phi - lambda_) / 2) * sin(theta / 2), exp(i * (phi + lambda_) / 2) * cos(theta / 2)]
    ])


def U_theta_phi_alpha(*, theta, phi, alpha) -> Matrix:
    """
    Parametrization used in "Dlaczego w dylemta więźnia warto grać kwantowo?" by Marek Szopa.
    https://www.ue.katowice.pl/fileadmin/_migrated/content_uploads/11_M.Szopa__Dlaczego_w_dylemat_wieznia....pdf
    :param theta: [0, PI]
    :param phi: [-PI, PI]
    :param alpha: [-PI, PI]
    :return:
    """
    return Matrix([
        [exp(-i * phi) * cos(theta / 2), exp(i * alpha) * sin(theta / 2)],
        [-exp(-i * alpha) * sin(theta / 2), exp(i * phi) * cos(theta / 2)]
    ])


def U_Eisert_Wilkens_Lewenstein(*, theta, phi) -> Matrix:
    """
    Original parametrization from "Quantum Games and Quantum Strategies"
    by Jens Eisert, Martin Wilkens, and Maciej Lewenstein (DOI:10.1103/PhysRevLett.83.3077).
    https://arxiv.org/pdf/quant-ph/9806088.pdf
    :param theta: [0, PI]
    :param phi: [0, PI/2]
    :return:
    """
    return Matrix([
        [exp(i * phi) * cos(theta / 2), sin(theta / 2)],
        [-sin(theta / 2), exp(-i * phi) * cos(theta / 2)]
    ])


def U_Frackiewicz_Pykacz(*, theta, phi) -> Matrix:
    """
    Full SU(2) parametrization from "Quantum Games with Strategies Induced by Basis Change Rules"
    by Piotr Frąckiewicz and Jarosław Pykacz (DOI:10.1007/s10773-017-3423-6).
    https://link.springer.com/content/pdf/10.1007%2Fs10773-017-3423-6.pdf
    :param theta: [0, PI]
    :param phi: [0, 2*PI]
    :return:
    """
    return Matrix([
        [exp(i * phi) * cos(theta / 2), i * exp(i * phi) * sin(theta / 2)],
        [i * exp(-i * phi) * sin(theta / 2), exp(-i * phi) * cos(theta / 2)]
    ])


def U_IBM(*, theta, phi, lambda_) -> Matrix:
    """
    General single-qubit quantum gate as defined in Qiskit textbook by IBM.
    https://qiskit.org/textbook/ch-states/single-qubit-gates.html#7.-The-U-gate--
    :param theta:
    :param phi:
    :param lambda_:
    :return:
    """
    return Matrix([
        [cos(theta / 2), -exp(i * lambda_) * sin(theta / 2)],
        [exp(i * phi) * sin(theta / 2), exp(i * (phi + lambda_)) * cos(theta / 2)]
    ])


def U(*args, **kwargs) -> Matrix:
    if args:
        raise Exception('Please use keyword arguments')
    if set(kwargs) in [{'theta', 'alpha'}, {'theta', 'alpha', 'beta'}]:
        return U_theta_alpha_beta(**kwargs)
    if set(kwargs) == {'theta', 'phi', 'lambda_'}:
        return U_theta_phi_lambda(**kwargs)
    if set(kwargs) == {'theta', 'phi'}:
        return U_Eisert_Wilkens_Lewenstein(**kwargs)
    raise Exception('Invalid parametrization')
