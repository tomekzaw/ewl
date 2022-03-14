import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from sympy.physics.quantum.qubit import Qubit

from ewl import EWL
from ewl.ibmq import EWL_IBMQ
from ewl.parametrizations import U_theta_alpha_beta


@pytest.fixture
def ewl_fixed() -> EWL:
    i = sp.I
    pi = sp.pi
    sqrt2 = sp.sqrt(2)
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U_theta_alpha_beta(theta=pi / 2, alpha=pi / 2, beta=0)
    bob = U_theta_alpha_beta(theta=0, alpha=0, beta=0)
    return EWL(psi, [alice, bob])


@pytest.fixture
def ewl_parametrized() -> EWL:
    i = sp.I
    sqrt2 = sp.sqrt(2)
    theta1, alpha1, beta1, theta2, alpha2, beta2 = sp.symbols('theta1 alpha1 beta1 theta2 alpha2 beta2', real=True)
    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U_theta_alpha_beta(theta=theta1, alpha=alpha1, beta=beta1)
    bob = U_theta_alpha_beta(theta=theta2, alpha=alpha2, beta=beta2)
    return EWL(psi, [alice, bob])


def test_init_fixed(ewl_fixed: EWL):
    EWL_IBMQ(ewl_fixed)


def test_init_parametrized(ewl_parametrized: EWL):
    with pytest.raises(Exception):
        EWL_IBMQ(ewl_parametrized)


def test_qc(ewl_fixed: EWL):
    i = 1j
    sqrt2 = np.sqrt(2)

    ewl_ibmq = EWL_IBMQ(ewl_fixed)

    j = Operator(np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ]))

    j_h = Operator(np.array([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, -i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, -i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ]))

    u_a = Operator(np.array([
        [i / sqrt2, i / sqrt2],
        [i / sqrt2, -i / sqrt2]
    ]))

    u_b = Operator(np.array([
        [1, 0],
        [0, 1],
    ]))

    qc = QuantumCircuit(2)
    qc.append(j, [0, 1])
    qc.barrier()
    qc.append(u_a, [0])
    qc.append(u_b, [1])
    qc.barrier()
    qc.append(j_h, [0, 1])
    qc.measure_all()

    assert ewl_ibmq.qc == qc
