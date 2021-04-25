import cmath
from functools import cached_property
from itertools import product
from math import log2
from typing import Optional, Sequence, Dict

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.ibmq.exceptions import IBMQAccountCredentialsNotFound, IBMQProviderError
from qiskit.quantum_info.operators import Operator
from qiskit.tools import job_monitor
from qiskit.visualization import plot_histogram  # noqa: F401
from sympy import Matrix
from sympy.physics.quantum import TensorProduct, Dagger
from sympy.physics.quantum.qubit import Qubit, qubit_to_matrix  # noqa: F401

try:
    IBMQ.load_account()
except IBMQAccountCredentialsNotFound:
    pass

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)


def number_of_qubits(psi) -> int:
    return int(log2(len(qubit_to_matrix(psi))))


def sympy_to_numpy_matrix(matrix: Matrix) -> np.array:
    return np.array(matrix).astype(complex)


def U_theta_alpha_beta(*, theta: complex, alpha: complex, beta: complex = 3 * cmath.pi / 2) -> Matrix:
    return Matrix([
        [sp.exp(i * alpha) * sp.cos(theta / 2), i * sp.exp(i * beta) * sp.sin(theta / 2)],
        [i * sp.exp(-i * beta) * sp.sin(theta / 2), sp.exp(-i * alpha) * sp.cos(theta / 2)]
    ])


def U_theta_phi_lambda(*, theta: complex, phi: complex, lambda_: complex) -> Matrix:
    return Matrix([
        [sp.exp(-i * (phi + lambda_) / 2) * sp.cos(theta / 2), -sp.exp(-i * (phi - lambda_) / 2) * sp.sin(theta / 2)],
        [sp.exp(i * (phi - lambda_) / 2) * sp.sin(theta / 2), sp.exp(i * (phi + lambda_) / 2) * sp.cos(theta / 2)]
    ])


def U(*args, **kwargs: complex) -> Matrix:
    if args:
        raise Exception('Please use keyword arguments')
    if set(kwargs) in [{'theta', 'alpha'}, {'theta', 'alpha', 'beta'}]:
        return U_theta_alpha_beta(**kwargs)
    if set(kwargs) == {'theta', 'phi', 'lambda_'}:
        return U_theta_phi_lambda(**kwargs)
    raise Exception('Invalid parametrization')


def J(psi, C: Matrix, D: Matrix) -> Matrix:
    return Matrix.hstack(*[
        TensorProduct(*base) @ qubit_to_matrix(psi)
        for base in product((C, D), repeat=number_of_qubits(psi))
    ])


class EWL:
    def __init__(self, psi, strategies: Sequence[Matrix], provider: Optional[AccountProvider] = None):
        assert number_of_qubits(psi) == len(strategies)

        self.psi = psi
        self.strategies = strategies

        if provider is None:
            try:
                self.provider = IBMQ.get_provider()
            except IBMQProviderError:
                raise RuntimeError('Please run this notebook on https://quantum-computing.ibm.com/lab')
        else:
            self.provider = provider

    @cached_property
    def number_of_players(self) -> int:
        return len(self.strategies)

    @cached_property
    def J(self) -> Matrix:
        C = U(theta=0, alpha=0, beta=0)
        D = U(theta=pi, alpha=0, beta=0)
        return J(self.psi, C, D)

    @cached_property
    def J_H(self) -> Matrix:
        return Dagger(self.J)

    def make_qc(self, *, measure: bool = True) -> QuantumCircuit:
        j = Operator(sympy_to_numpy_matrix(self.J))
        j_h = Operator(sympy_to_numpy_matrix(self.J_H))

        all_qbits = range(self.number_of_players)

        qc = QuantumCircuit(self.number_of_players)
        qc.append(j, all_qbits)
        qc.barrier()

        for qbit, strategy in enumerate(self.strategies):
            qc.append(Operator(sympy_to_numpy_matrix(strategy)), [qbit])

        qc.barrier()
        qc.append(j_h, all_qbits)

        if measure:
            qc.measure_all()

        return qc

    @cached_property
    def qc(self) -> QuantumCircuit:
        return self.make_qc(measure=True)

    def draw(self):
        return self.qc.draw('mpl')

    def draw_transpiled(self, backend_name: str, *, optimization_level: int = 3):
        backend = self.provider.get_backend(backend_name)
        transpiled_qc = transpile(self.qc, backend, optimization_level=optimization_level)
        return transpiled_qc.draw('mpl')

    def calculate_probs(self) -> Matrix:
        phi = self.J_H @ TensorProduct(*self.strategies) @ qubit_to_matrix(self.psi)
        return phi.multiply_elementwise(phi.conjugate())

    def simulate_probs(self, backend_name: str = 'statevector_simulator') -> Dict[str, float]:
        circ = self.make_qc(measure=False)
        simulator = Aer.get_backend(backend_name)
        return execute(circ, simulator).result().get_counts()

    def simulate_counts(self, backend_name: str = 'qasm_simulator') -> Dict[str, int]:
        simulator = Aer.get_backend(backend_name)
        return execute(self.qc, simulator).result().get_counts()

    def run(self, backend_name: str = 'least_busy', *, optimization_level: int = 3) -> Dict[str, int]:
        if backend_name == 'least_busy':
            small_devices = self.provider.backends(
                filters=lambda x: x.configuration().n_qubits >= self.number_of_players
                                  and not x.configuration().simulator and x.status().operational)  # noqa: W503, E131
            backend = least_busy(small_devices)
        else:
            backend = self.provider.get_backend(backend_name)

        job = execute(self.qc, backend, optimization_level=optimization_level)
        job_monitor(job)
        return job.result().get_counts()
