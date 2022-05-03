import sympy as sp
from sympy.plotting import plot3d
from sympy.physics.quantum.qubit import Qubit

from ewl.mixed import MixedEWL, MixedStrategy
from ewl.parametrizations import U_theta_phi_alpha

if __name__ == '__main__':
    # Example from "Dlaczego w dylemat więźnia warto grać kwantowo?" by Marek Szopa.
    # https://www.ue.katowice.pl/fileadmin/_migrated/content_uploads/11_M.Szopa__Dlaczego_w_dylemat_wieznia....pdf

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

    alice = MixedStrategy([
        (sp.cos(gamma_A / 2) ** 2, A_hat),
        (sp.sin(gamma_A / 2) ** 2, A_hat_prim),
    ])
    bob = MixedStrategy([
        (sp.cos(gamma_B / 2) ** 2, B_hat),
        (sp.sin(gamma_B / 2) ** 2, B_hat_prim),
    ])

    mixed_ewl = MixedEWL(psi=psi, C=C, D=D, players=[alice, bob], payoff_matrix=payoff_matrix)

    mixed_ewl_fixed = mixed_ewl.fix(theta_A=pi / 2, phi_A=0, alpha_A=0)

    alice_payoff_function = mixed_ewl_fixed.payoff_function(player=0)

    plot3d(alice_payoff_function, (gamma_A, 0, pi), (gamma_B, 0, pi))
