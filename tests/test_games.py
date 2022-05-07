import sympy as sp
from sympy import Matrix

from ewl import EWL
from ewl.games import make_Quantum_Prisoners_Dilemma_Eisert_Wilkens_Lewenstein, make_Quantum_Prisoners_Dilemma_Chen, \
    make_Quantum_Prisoners_Dilemma_Szopa, make_Quantum_Prisoners_Dilemma_theta_alpha_beta, make_Quantum_Prisoners_Dilemma_Frackiewicz_Pykacz
from ewl.parametrizations import U_Eisert_Wilkens_Lewenstein

i = sp.I
pi = sp.pi
sin = sp.sin
cos = sp.cos

theta_A, phi_A, alpha_A, beta_A = sp.symbols('theta_A phi_A alpha_A beta_A', real=True)
theta_B, phi_B, alpha_B, beta_B = sp.symbols('theta_B phi_B alpha_B beta_B', real=True)


def check_classical_payoffs(ewl: EWL) -> None:
    # from Fig. 1 in https://arxiv.org/pdf/quant-ph/0004076.pdf
    assert ewl.play(ewl.C, ewl.C).payoffs() == (3, 3)
    assert ewl.play(ewl.C, ewl.D).payoffs() == (0, 5)
    assert ewl.play(ewl.D, ewl.C).payoffs() == (5, 0)
    assert ewl.play(ewl.D, ewl.D).payoffs() == (1, 1)


def test_make_Quantum_Prisoners_Dilemma_Eisert_Wilkens_Lewenstein() -> None:
    ewl = make_Quantum_Prisoners_Dilemma_Eisert_Wilkens_Lewenstein()

    assert ewl.C == Matrix([[1, 0], [0, 1]])
    assert ewl.D == Matrix([[0, 1], [-1, 0]])

    check_classical_payoffs(ewl)

    # miracle move from Eq. 8 in https://arxiv.org/pdf/quant-ph/0004076.pdf
    Q = U_Eisert_Wilkens_Lewenstein(theta=0, phi=pi / 2)
    assert Q == Matrix([[i, 0], [0, -i]])
    assert ewl.play(Q, Q).payoffs() == (3, 3)


def test_make_Quantum_Prisoners_Dilemma_Chen() -> None:
    ewl = make_Quantum_Prisoners_Dilemma_Chen()

    assert ewl.C == Matrix([[1, 0], [0, 1]])
    assert ewl.D == Matrix([[0, i], [i, 0]])

    check_classical_payoffs(ewl)

    # from page 7 in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1344&rep=rep1&type=pdf (corrected)
    assert ewl.probs().equals(Matrix([
        (cos(theta_A / 2) * cos(theta_B / 2) * cos(phi_A + phi_B) - sin(theta_A / 2) * sin(theta_B / 2) * sin(alpha_A + alpha_B)) ** 2,
        (sin(theta_A / 2) * cos(theta_B / 2) * cos(alpha_A + phi_B) + cos(theta_A / 2) * sin(theta_B / 2) * sin(phi_A + alpha_B)) ** 2,
        (sin(theta_A / 2) * cos(theta_B / 2) * sin(alpha_A + phi_B) + cos(theta_A / 2) * sin(theta_B / 2) * cos(phi_A + alpha_B)) ** 2,
        (cos(theta_A / 2) * cos(theta_B / 2) * sin(phi_A + phi_B) - sin(theta_A / 2) * sin(theta_B / 2) * cos(alpha_A + alpha_B)) ** 2,
    ]))

    alice_payoff_function = ewl.payoff_function(player=0)
    bob_payoff_function = ewl.payoff_function(player=1)

    # Bob's best response 1 (corrected)
    assert sp.simplify(bob_payoff_function.subs({
        theta_B: theta_A + pi,
        phi_B: -alpha_A,
        alpha_B: -phi_A - pi / 2,
    })) == 5

    # Bob's best response 2 (corrected)
    assert sp.simplify(bob_payoff_function.subs({
        theta_B: pi - theta_A,
        phi_B: -alpha_A,
        alpha_B: -phi_A + pi / 2,
    })) == 5

    # Alice's best response 1 (corrected)
    assert sp.simplify(alice_payoff_function.subs({
        theta_A: theta_B + pi,
        phi_A: -alpha_B,
        alpha_A: -phi_B - pi / 2,
    })) == 5

    # Alice's best response 2 (corrected)
    assert sp.simplify(alice_payoff_function.subs({
        theta_A: pi - theta_B,
        phi_A: -alpha_B,
        alpha_A: -phi_B + pi / 2,
    })) == 5


def test_make_Quantum_Prisoners_Dilemma_Szopa() -> None:
    ewl = make_Quantum_Prisoners_Dilemma_Szopa()

    assert ewl.C == Matrix([[1, 0], [0, 1]])
    assert ewl.D == Matrix([[0, 1], [-1, 0]])

    check_classical_payoffs(ewl)

    # from page 178 in https://www.ue.katowice.pl/fileadmin/_migrated/content_uploads/11_M.Szopa__Dlaczego_w_dylemat_wieznia....pdf (corrected)
    assert ewl.probs().equals(Matrix([
        (sin(theta_A / 2) * sin(theta_B / 2) * sin(alpha_A + alpha_B) - cos(theta_A / 2) * cos(theta_B / 2) * cos(phi_A + phi_B)) ** 2,
        (sin(theta_A / 2) * cos(theta_B / 2) * sin(alpha_A + phi_B) + sin(theta_B / 2) * cos(theta_A / 2) * cos(alpha_B + phi_A)) ** 2,
        (sin(theta_A / 2) * cos(theta_B / 2) * cos(alpha_A + phi_B) + sin(theta_B / 2) * cos(theta_A / 2) * sin(alpha_B + phi_A)) ** 2,
        (sin(theta_A / 2) * sin(theta_B / 2) * cos(alpha_A + alpha_B) - cos(theta_A / 2) * cos(theta_B / 2) * sin(phi_A + phi_B)) ** 2,
    ]))

    alice_payoff_function = ewl.payoff_function(player=0)
    bob_payoff_function = ewl.payoff_function(player=1)

    # Bob's best response 1 (corrected)
    assert sp.simplify(bob_payoff_function.subs({
        theta_B: theta_A + pi,
        phi_B: -alpha_A - pi / 2,
        alpha_B: -phi_A,
    })) == 5

    # Bob's best response 2 (corrected)
    assert sp.simplify(bob_payoff_function.subs({
        theta_B: pi - theta_A,
        phi_B: -alpha_A + pi / 2,
        alpha_B: -phi_A,
    })) == 5

    # Alice's best response 1 (corrected)
    assert sp.simplify(alice_payoff_function.subs({
        theta_A: theta_B + pi,
        phi_A: -alpha_B - pi / 2,
        alpha_A: -phi_B,
    })) == 5

    # Alice's best response 2 (corrected)
    assert sp.simplify(alice_payoff_function.subs({
        theta_A: pi - theta_B,
        phi_A: -alpha_B + pi / 2,
        alpha_A: -phi_B,
    })) == 5


def test_make_Quantum_Prisoners_Dilemma_theta_alpha_beta() -> None:
    ewl = make_Quantum_Prisoners_Dilemma_theta_alpha_beta()

    assert ewl.C == Matrix([[1, 0], [0, 1]])
    assert ewl.D == Matrix([[0, i], [i, 0]])

    check_classical_payoffs(ewl)

    # from Eq. 7 in https://link.springer.com/content/pdf/10.1007%2Fs10773-017-3423-6.pdf (corrected)
    assert ewl.probs().equals(Matrix([
        (cos(alpha_A + alpha_B) * cos(theta_A / 2) * cos(theta_B / 2) + sin(beta_A + beta_B) * sin(theta_A / 2) * sin(theta_B / 2)) ** 2,
        (cos(alpha_A - beta_B) * cos(theta_A / 2) * sin(theta_B / 2) + sin(alpha_B - beta_A) * sin(theta_A / 2) * cos(theta_B / 2)) ** 2,
        (sin(alpha_A - beta_B) * cos(theta_A / 2) * sin(theta_B / 2) + cos(alpha_B - beta_A) * sin(theta_A / 2) * cos(theta_B / 2)) ** 2,
        (sin(alpha_A + alpha_B) * cos(theta_A / 2) * cos(theta_B / 2) - cos(beta_A + beta_B) * sin(theta_A / 2) * sin(theta_B / 2)) ** 2,
    ]))


def test_make_Quantum_Prisoners_Dilemma_Frackiewicz_Pykacz() -> None:
    ewl = make_Quantum_Prisoners_Dilemma_Frackiewicz_Pykacz()

    assert ewl.C == Matrix([[1, 0], [0, 1]])
    assert ewl.D == Matrix([[0, i], [i, 0]])

    check_classical_payoffs(ewl)

    # from Eq. 10 in https://mdpi-res.com/d_attachment/entropy/entropy-23-00506/article_deploy/entropy-23-00506-v2.pdf
    assert ewl.probs().equals(Matrix([
        (cos(theta_A / 2) * cos(theta_B / 2) * cos(phi_A + phi_B) + sin(theta_A / 2) * sin(theta_B / 2) * sin(phi_A + phi_B)) ** 2,
        (cos(theta_A / 2) * sin(theta_B / 2) * cos(phi_A - phi_B) - sin(theta_A / 2) * cos(theta_B / 2) * sin(phi_A - phi_B)) ** 2,
        (cos(theta_A / 2) * sin(theta_B / 2) * sin(phi_A - phi_B) + sin(theta_A / 2) * cos(theta_B / 2) * cos(phi_A - phi_B)) ** 2,
        (cos(theta_A / 2) * cos(theta_B / 2) * sin(phi_A + phi_B) - sin(theta_A / 2) * sin(theta_B / 2) * cos(phi_A + phi_B)) ** 2,
    ]))
