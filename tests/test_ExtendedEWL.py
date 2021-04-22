import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from sympy import Matrix
from sympy.physics.quantum.qubit import Qubit

from ewl import U, ExtendedEWL


@pytest.fixture
def ewl() -> ExtendedEWL:
    i = sp.I
    pi = sp.pi
    sqrt2 = sp.sqrt(2)

    psi = (Qubit('00') + i * Qubit('11')) / sqrt2
    alice = U(theta=pi / 2, alpha=pi / 2, beta=0)
    bob = U(theta=0, alpha=0, beta=0)

    return ExtendedEWL(psi, [alice, bob])


def test_j(ewl: ExtendedEWL):
    i = sp.I
    sqrt2 = sp.sqrt(2)

    assert ewl.j == Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])


def test_j_h(ewl: ExtendedEWL):
    i = sp.I
    sqrt2 = sp.sqrt(2)

    assert ewl.j_h == Matrix([
        [1 / sqrt2, 0, 0, -i / sqrt2],
        [0, -i / sqrt2, -1 / sqrt2, 0],
        [0, -1 / sqrt2, -i / sqrt2, 0],
        [i / sqrt2, 0, 0, -1 / sqrt2],
    ])


def test_qc(ewl: ExtendedEWL):
    i = 1j
    sqrt2 = np.sqrt(2)

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

    assert ewl.qc == qc
