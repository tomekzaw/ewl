import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.extensions.unitary import UnitaryGate
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info.operators import Operator
from sympy import Matrix
from sympy.physics.quantum.qubit import Qubit

from ewl import EWL
from ewl.ibmq import EWL_IBMQ
from ewl.parametrizations import U_theta_alpha_beta

i = sp.I

C = Matrix([[1, 0], [0, 1]])
D = Matrix([[0, i], [i, 0]])


@pytest.fixture
def ewl_fixed() -> EWL:
    i = sp.I
    pi = sp.pi
    sqrt2 = sp.sqrt(2)
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U_theta_alpha_beta(theta=pi / 2, alpha=pi / 2, beta=0)
    bob = U_theta_alpha_beta(theta=0, alpha=0, beta=0)
    return EWL(psi=psi, C=C, D=D, players=[alice, bob])


@pytest.fixture
def ewl_parametrized() -> EWL:
    i = sp.I
    sqrt2 = sp.sqrt(2)
    theta1, alpha1, beta1, theta2, alpha2, beta2 = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2', real=True)
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U_theta_alpha_beta(theta=theta1, alpha=alpha1, beta=beta1)
    bob = U_theta_alpha_beta(theta=theta2, alpha=alpha2, beta=beta2)
    return EWL(psi=psi, C=C, D=D, players=[alice, bob])


def test_init_fixed(ewl_fixed: EWL):
    EWL_IBMQ(ewl_fixed)


def test_init_noise_model(ewl_fixed: EWL):
    EWL_IBMQ(ewl_fixed, noise_model=NoiseModel())


def test_init_parametrized(ewl_parametrized: EWL):
    with pytest.raises(Exception):
        EWL_IBMQ(ewl_parametrized)


def test_qc(ewl_fixed: EWL):
    i = 1j
    sqrt2 = np.sqrt(2)

    ewl_ibmq = EWL_IBMQ(ewl_fixed)

    J = UnitaryGate(Operator(np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])), label='$J$')

    J_H = UnitaryGate(Operator(np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, -i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, -i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])), label='$J^\\dagger$')

    U_0 = UnitaryGate(Operator(np.array([
        [i / sqrt2, i / sqrt2],
        [i / sqrt2, -i / sqrt2]
    ])), label='$U_{0}$')

    U_1 = UnitaryGate(Operator(np.array([
        [1, 0],
        [0, 1],
    ])), label='$U_{1}$')

    qc = QuantumCircuit(2)
    qc.append(J, [0, 1])
    qc.barrier()
    qc.append(U_0, [0])
    qc.append(U_1, [1])
    qc.barrier()
    qc.append(J_H, [0, 1])
    qc.measure_all()

    assert ewl_ibmq.qc == qc
