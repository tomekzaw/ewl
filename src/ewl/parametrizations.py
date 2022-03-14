import sympy as sp
from sympy import Matrix

i = sp.I
pi = sp.pi
sin = sp.sin
cos = sp.cos
exp = sp.exp


def U_theta_alpha_beta(*, theta, alpha, beta) -> Matrix:
    """
    Full SU(2) parametrization from "A quantum approach to twice-repeated 2×2 game"
    by Katarzyna Rycerz and Piotr Frąckiewicz (DOI:10.1007/s11128-020-02743-0).
    https://www.researchgate.net/publication/343293505_A_quantum_approach_to_twice-repeated_2times_2_game
    :param theta:
    :param alpha:
    :param beta:
    :return:
    """
    return Matrix([
        [exp(i * alpha) * cos(theta / 2), i * exp(i * beta) * sin(theta / 2)],
        [i * exp(-i * beta) * sin(theta / 2), exp(-i * alpha) * cos(theta / 2)]
    ])


def U_theta_gamma_delta(*, theta, gamma, delta) -> Matrix:
    """
    Full SU(2) parametrization from "A quantum approach to twice-repeated 2×2 game"
    by Katarzyna Rycerz and Piotr Frąckiewicz (DOI:10.1007/s11128-020-02743-0).
    https://www.researchgate.net/publication/343293505_A_quantum_approach_to_twice-repeated_2times_2_game
    :param theta:
    :param gamma:
    :param delta:
    :return:
    """
    return Matrix([
        [exp(i * (gamma + delta)) * cos(theta / 2), i * exp(i * (gamma - delta)) * sin(theta / 2)],
        [i * exp(-i * (gamma - delta)) * sin(theta / 2), exp(-i * (gamma + delta)) * cos(theta / 2)]
    ])


def U_theta_phi_lambda(*, theta, phi, lambda_) -> Matrix:
    """
    Full SU(2) parametrization.
    :param theta:
    :param phi:
    :param lambda_:
    :return:
    """
    return Matrix([
        [exp(-i * (phi + lambda_) / 2) * cos(theta / 2), -exp(-i * (phi - lambda_) / 2) * sin(theta / 2)],
        [exp(i * (phi - lambda_) / 2) * sin(theta / 2), exp(i * (phi + lambda_) / 2) * cos(theta / 2)]
    ])


def U_theta_phi_alpha(*, theta, phi, alpha) -> Matrix:
    """
    Parametrization used in "Dlaczego w dylemat więźnia warto grać kwantowo?" by Marek Szopa.
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
