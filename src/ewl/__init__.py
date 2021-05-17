import warnings
from functools import cached_property, reduce
from itertools import product
from math import log2
from operator import add
from typing import Sequence, Dict, Set, Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import MatplotlibDeprecationWarning
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.ibmq.exceptions import IBMQAccountCredentialsNotFound, IBMQProviderError
from qiskit.quantum_info.operators import Operator
from qiskit.tools import job_monitor
from qiskit.visualization import plot_histogram  # noqa: F401
from sympy import init_printing, Matrix, Array
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.qubit import Qubit, qubit_to_matrix  # noqa: F401

try:
    from functools import cache  # Python 3.9
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)

init_printing()
warnings.simplefilter('ignore', category=MatplotlibDeprecationWarning)

try:
    IBMQ.load_account()
except IBMQAccountCredentialsNotFound:
    pass

i = sp.I
pi = sp.pi
sqrt2 = sp.sqrt(2)


def number_of_qubits(psi) -> int:
    return int(log2(len(qubit_to_matrix(psi))))


def convert_exp_to_trig(expr):
    return expr.rewrite(sp.sin).simplify()


def amplitude_to_prob(expr):
    return sp.Abs(expr) ** 2


def sympy_to_numpy_matrix(matrix: Matrix) -> np.array:
    return np.array(matrix).astype(complex)


def U_theta_alpha_beta(*, theta, alpha, beta=3 * sp.pi / 2) -> Matrix:
    return Matrix([
        [sp.exp(i * alpha) * sp.cos(theta / 2), i * sp.exp(i * beta) * sp.sin(theta / 2)],
        [i * sp.exp(-i * beta) * sp.sin(theta / 2), sp.exp(-i * alpha) * sp.cos(theta / 2)]
    ])


def U_theta_phi_lambda(*, theta, phi, lambda_) -> Matrix:
    return Matrix([
        [sp.exp(-i * (phi + lambda_) / 2) * sp.cos(theta / 2), -sp.exp(-i * (phi - lambda_) / 2) * sp.sin(theta / 2)],
        [sp.exp(i * (phi - lambda_) / 2) * sp.sin(theta / 2), sp.exp(i * (phi + lambda_) / 2) * sp.cos(theta / 2)]
    ])


def U_theta_phi(*, theta, phi) -> Matrix:
    # original parametrization from "Quantum Games and Quantum Strategies" by J. Eisert, M. Wilkens, M. Lewenstein
    return Matrix([
        [sp.exp(i * phi) * sp.cos(theta / 2), sp.sin(theta / 2)],
        [-sp.sin(theta / 2), sp.exp(-i * phi) * sp.cos(theta / 2)]
    ])


def U(*args, **kwargs) -> Matrix:
    if args:
        raise Exception('Please use keyword arguments')
    if set(kwargs) in [{'theta', 'alpha'}, {'theta', 'alpha', 'beta'}]:
        return U_theta_alpha_beta(**kwargs)
    if set(kwargs) == {'theta', 'phi', 'lambda_'}:
        return U_theta_phi_lambda(**kwargs)
    if set(kwargs) == {'theta', 'phi'}:
        return U_theta_phi(**kwargs)
    raise Exception('Invalid parametrization')


def J(psi, C: Matrix, D: Matrix) -> Matrix:
    return Matrix.hstack(*[
        TensorProduct(*base) @ qubit_to_matrix(psi)
        for base in product((C, D), repeat=number_of_qubits(psi))
    ])


class EWL:
    def __init__(self, psi, strategies: Sequence[Matrix], payoff_matrix: Optional[Array] = None, noise_model: Optional[NoiseModel] = None):
        assert number_of_qubits(psi) == len(strategies), 'Number of qubits and strategies must be equal'

        if payoff_matrix is not None:
            assert payoff_matrix.rank() == len(strategies) + 1, 'Invalid number of dimensions of payoff matrix'
            assert payoff_matrix.shape == (len(strategies),) + (2,) * len(strategies), 'Invalid shape of payoff matrix'

        self.psi = psi
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix
        self.noise_model = noise_model

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
        return self.J.H

    @cache
    def amplitudes(self, *, simplify: bool = True) -> Matrix:
        ampl = self.J_H @ TensorProduct(*self.strategies) @ qubit_to_matrix(self.psi)
        if simplify:
            ampl = ampl.applyfunc(convert_exp_to_trig)
        return ampl

    @cache
    def probs(self, *, simplify: bool = True) -> Matrix:
        return self.amplitudes(simplify=simplify).applyfunc(amplitude_to_prob)

    @cache
    def payoff_function(self, *, player: Optional[int], simplify: bool = True):
        if self.payoff_matrix is None:
            raise Exception('Payoff matrix not defined')

        if player is not None:
            assert player < self.number_of_players, 'Invalid number of player'

        probs = self.probs(simplify=simplify)
        payoff_matrix = self.payoff_matrix[player] if player is not None else reduce(add, self.payoff_matrix)
        return sum(
            probs[i] * payoff_matrix[idx]
            for i, idx in enumerate(product(range(2), repeat=self.number_of_players))
        )

    def plot_payoff_function(self, *, player: Optional[int],
                             x: sp.Symbol, x_min, x_max, y: sp.Symbol, y_min, y_max, n: int = 20,
                             figsize=(8, 8), cmap=plt.cm.coolwarm, **kwargs) -> plt.Figure:
        f = sp.lambdify([x, y], self.payoff_function(player=player))

        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(xs, ys)
        Z = f(X, Y)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cmap, antialiased=True)
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
               xlabel=f'${sp.latex(x)}$', ylabel=f'${sp.latex(y)}$',
               zlabel='expected payoff', **kwargs)
        return fig

    @cached_property
    def params(self) -> Set[sp.Symbol]:
        return self.psi.atoms(sp.Symbol).union(*(
            strategy.atoms(sp.Symbol)
            for strategy in self.strategies
        ))

    def fix(self, **kwargs):
        params = {sp.Symbol(k): v for k, v in kwargs.items()}
        psi = self.psi.subs(params)
        strategies = [strategy.subs(params) for strategy in self.strategies]
        payoff_matrix = self.payoff_matrix.subs(params) if self.payoff_matrix is not None else None
        noise_model = self.noise_model
        return type(self)(psi, strategies, payoff_matrix, noise_model)

    @cached_property
    def provider(self) -> AccountProvider:
        try:
            return IBMQ.get_provider()
        except IBMQProviderError:
            raise RuntimeError('Please run this notebook on https://quantum-computing.ibm.com/lab '
                               'or save account token using IBMQ.save_account function')

    def make_qc(self, *, measure: bool = True) -> QuantumCircuit:
        if self.params:
            raise Exception('Please provide values for the following parameters: ' + ', '.join(map(str, self.params)))

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

    def simulate_probs(self, backend_name: str = 'statevector_simulator') -> Dict[str, float]:
        circ = self.make_qc(measure=False)
        simulator = Aer.get_backend(backend_name)
        return execute(circ, simulator).result().get_counts()

    def simulate_counts(self) -> Dict[str, int]:
        simulator = AerSimulator(noise_model=self.noise_model)
        circ = transpile(self.qc, simulator) if self.noise_model is not None else self.qc
        result = simulator.run(circ).result()
        return result.get_counts()

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
