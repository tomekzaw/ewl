import sympy as sp
from sympy import Array
from sympy.physics.quantum.qubit import Qubit

from ewl import EWL
from ewl.parametrizations import U_Eisert_Wilkens_Lewenstein, U_Frackiewicz_Pykacz, U_theta_phi_alpha, U_theta_alpha_beta

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)

theta_A, phi_A, alpha_A, beta_A = sp.symbols('theta_A phi_A alpha_A beta_A', real=True)
theta_B, phi_B, alpha_B, beta_B = sp.symbols('theta_B phi_B alpha_B beta_B', real=True)

psi = 1 / sqrt2 * Qubit('00') + i / sqrt2 * Qubit('11')

payoff_matrix = Array([
    [
        [3, 0],
        [5, 1],
    ],
    [
        [3, 5],
        [0, 1],
    ],
])


def make_Quantum_Prisoners_Dilemma_Eisert_Wilkens_Lewenstein() -> EWL:
    """
    Quantum Prisoner's Dilemma with original EWL parametrization
    from "Quantum Games and Quantum Strategies" by Jens Eisert, Martin Wilkens and Maciej Lewenstein
    (https://arxiv.org/pdf/quant-ph/9806088.pdf)
    """
    alice = U_Eisert_Wilkens_Lewenstein(theta=theta_A, phi=phi_A)
    bob = U_Eisert_Wilkens_Lewenstein(theta=theta_B, phi=phi_B)

    C = U_Eisert_Wilkens_Lewenstein(theta=0, phi=0)
    D = U_Eisert_Wilkens_Lewenstein(theta=pi, phi=0)

    return EWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)


def make_Quantum_Prisoners_Dilemma_Chen() -> EWL:
    """
    Quantum Prisoner's Dilemma with full SU(2) parametrization
    from "How Well Do People Play a Quantum Prisoner's Dilemma?" by Kay-Yut Chen and Tad Hogg
    (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1344&rep=rep1&type=pdf)
    """
    alice = U_theta_phi_alpha(theta=theta_A, phi=phi_A, alpha=alpha_A)
    bob = U_theta_phi_alpha(theta=theta_B, phi=phi_B, alpha=alpha_B)

    C = U_theta_phi_alpha(theta=0, phi=0, alpha=0)
    D = U_theta_phi_alpha(theta=pi, phi=0, alpha=pi / 2)

    return EWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)


def make_Quantum_Prisoners_Dilemma_Szopa() -> EWL:
    """
    Quantum Prisoner's Dilemma with full SU(2) parametrization
    from "Dlaczego w dylemat więźnia warto grać kwantowo?" by Marek Szopa
    (https://www.ue.katowice.pl/fileadmin/_migrated/content_uploads/11_M.Szopa__Dlaczego_w_dylemat_wieznia....pdf)
    """
    alice = U_theta_phi_alpha(theta=theta_A, phi=phi_A, alpha=alpha_A)
    bob = U_theta_phi_alpha(theta=theta_B, phi=phi_B, alpha=alpha_B)

    C = U_theta_phi_alpha(theta=0, phi=0, alpha=0)
    D = U_theta_phi_alpha(theta=pi, phi=0, alpha=0)

    return EWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)


def make_Quantum_Prisoners_Dilemma_theta_alpha_beta() -> EWL:
    """
    Quantum Prisoner's Dilemma with full SU(2) parametrization
    from "Quantum games with strategies induced by basis change rules" by Piotr Frąckiewicz and Jarosław Pykacz
    (https://www.researchgate.net/publication/317754594_Quantum_Games_with_Strategies_Induced_by_Basis_Change_Rules/fulltext/59601fed0f7e9b8194fc0d96/Quantum-Games-with-Strategies-Induced-by-Basis-Change-Rules.pdf)
    """
    alice = U_theta_alpha_beta(theta=theta_A, alpha=alpha_A, beta=beta_A)
    bob = U_theta_alpha_beta(theta=theta_B, alpha=alpha_B, beta=beta_B)

    C = U_theta_alpha_beta(theta=0, alpha=0, beta=0)
    D = U_theta_alpha_beta(theta=pi, alpha=0, beta=0)

    return EWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)


def make_Quantum_Prisoners_Dilemma_Frackiewicz_Pykacz() -> EWL:
    """
    Quantum Prisoner's Dilemma with Frąckiewicz-Pykacz parametrization
    from "Quantum games with strategies induced by basis change rules" by Piotr Frąckiewicz and Jarosław Pykacz
    (https://www.researchgate.net/publication/317754594_Quantum_Games_with_Strategies_Induced_by_Basis_Change_Rules/fulltext/59601fed0f7e9b8194fc0d96/Quantum-Games-with-Strategies-Induced-by-Basis-Change-Rules.pdf)
    """
    alice = U_Frackiewicz_Pykacz(theta=theta_A, phi=phi_A)
    bob = U_Frackiewicz_Pykacz(theta=theta_B, phi=phi_B)

    C = U_Frackiewicz_Pykacz(theta=0, phi=0)
    D = U_Frackiewicz_Pykacz(theta=pi, phi=0)

    return EWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)
