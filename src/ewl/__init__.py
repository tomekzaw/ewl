import cmath
from functools import cached_property
from itertools import product
from math import log2
from typing import Optional, Sequence

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
from sympy.physics.quantum.qubit import qubit_to_matrix

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


class ExtendedEWL:
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
    def j(self) -> Matrix:
        C = U(theta=0, alpha=0, beta=0)
        D = U(theta=pi, alpha=0, beta=0)
        return J(self.psi, C, D)

    @cached_property
    def j_h(self) -> Matrix:
        return Dagger(self.j)

    @cached_property
    def qc(self) -> QuantumCircuit:
        j = Operator(sympy_to_numpy_matrix(self.j))
        j_h = Operator(sympy_to_numpy_matrix(self.j_h))

        all_qbits = list(range(self.number_of_players))

        qc = QuantumCircuit(self.number_of_players)
        qc.append(j, all_qbits)
        qc.barrier()

        for qbit, strategy in enumerate(self.strategies):
            qc.append(Operator(sympy_to_numpy_matrix(strategy)), [qbit])

        qc.barrier()
        qc.append(j_h, all_qbits)
        qc.measure_all()

        return qc

    def draw(self):
        return self.qc.draw('mpl')

    def draw_transpiled(self, backend_name: str, *, optimization_level: int = 3):
        backend = self.provider.get_backend(backend_name)
        transpiled_circ = transpile(self.qc, backend, optimization_level=optimization_level)
        return transpiled_circ.draw('mpl')

    def simulate(self, backend_name: str = 'qasm_simulator'):
        simulator = Aer.get_backend(backend_name)
        result = execute(self.qc, simulator).result()
        counts_simulated = result.get_counts(self.qc)
        return counts_simulated

    def run(self, backend_name: str = 'least_busy', *, optimization_level: int = 3):
        if backend_name == 'least_busy':
            small_devices = self.provider.backends(filters=lambda x: 2 <= x.configuration().n_qubits <= 5
                                                                     and not x.configuration().simulator  # noqa: E127, W503
                                                                     and x.status().operational)  # noqa: E127, W503
            backend = least_busy(small_devices)
        else:
            backend = self.provider._backends[backend_name]  # TODO: improve

        job = execute(self.qc, backend, optimization_level=optimization_level)
        job_monitor(job)
        results = job.result()
        counts_quantum = results.get_counts()
        return counts_quantum
