from cmath import sqrt, pi, sin, cos, exp
from itertools import product
from functools import cached_property, reduce
from math import log2
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.ibmq.exceptions import IBMQAccountCredentialsNotFound, IBMQProviderError
from qiskit.quantum_info.operators import Operator
from qiskit.tools import job_monitor
from qiskit.visualization import plot_histogram  # noqa: F401

try:
    IBMQ.load_account()
except IBMQAccountCredentialsNotFound:
    pass

i = 1j
sqrt2 = sqrt(2)


def ket(base_state: str, /) -> np.array:
    qbits = len(base_state)
    assert qbits > 0, 'Number of qbits must be at least 1'
    assert set(base_state) <= {'0', '1'}, 'Base state must consist of only 0s and 1s'

    vector = np.zeros(2 ** qbits)
    vector[int(base_state, 2)] = 1
    return vector


def is_unit_vector(vec: np.array) -> bool:
    return np.isclose(np.linalg.norm(vec), 1)


def number_of_qbits(psi: np.array) -> int:
    n = psi.shape[0]
    assert (n & (n - 1) == 0) and n != 0, 'Vector size must be a power of two'
    return int(log2(n))


def U_theta_alpha_beta(*, theta: complex, alpha: complex, beta: complex = 3 * pi / 2) -> np.array:
    return np.array([
        [exp(i * alpha) * cos(theta / 2), i * exp(i * beta) * sin(theta / 2)],
        [i * exp(-i * beta) * sin(theta / 2), exp(-i * alpha) * cos(theta / 2)]
    ])


def U_theta_phi_lambda(*, theta: complex, phi: complex, lambda_: complex) -> np.array:
    return np.array([
        [exp(-i * (phi + lambda_) / 2) * cos(theta / 2), -exp(-i * (phi - lambda_) / 2) * sin(theta / 2)],
        [exp(i * (phi - lambda_) / 2) * sin(theta / 2), exp(i * (phi + lambda_) / 2) * cos(theta / 2)]
    ])


def U(*args, **kwargs: complex) -> np.array:
    if args:
        raise Exception('Please use keyword arguments')
    if set(kwargs) in [{'theta', 'alpha'}, {'theta', 'alpha', 'beta'}]:
        return U_theta_alpha_beta(**kwargs)
    if set(kwargs) == {'theta', 'phi', 'lambda_'}:
        return U_theta_phi_lambda(**kwargs)
    raise Exception('Invalid parametrization')


def J(psi: np.array, C: np.array, D: np.array) -> np.array:
    return np.column_stack([
        np.dot(reduce(np.kron, base), psi)
        for base in product((C, D), repeat=number_of_qbits(psi))
    ])


class ExtendedEWL:
    def __init__(self, psi: np.array, strategies: Sequence[np.array], provider: Optional[AccountProvider] = None):
        assert is_unit_vector(psi), 'Initial state must be a unit vector'
        assert len(strategies) == number_of_qbits(psi), 'Number of strategies must be equal to number of qbits'

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
    def j(self) -> np.array:
        C = U(theta=0, alpha=0, beta=0)
        D = U(theta=pi, alpha=0, beta=0)
        return J(self.psi, C, D)

    @cached_property
    def qc(self) -> QuantumCircuit:
        j = Operator(self.j)
        j_h = Operator(np.conjugate(self.j).T)
        all_qbits = list(range(self.number_of_players))

        qc = QuantumCircuit(self.number_of_players)
        qc.append(j, all_qbits)
        qc.barrier()

        for qbit, strategy in enumerate(self.strategies):
            u_a = Operator(strategy)
            qc.append(u_a, [qbit])

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
