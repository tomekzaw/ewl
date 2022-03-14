from __future__ import annotations

from functools import cached_property, reduce
from itertools import product
from operator import add
from typing import Optional, Sequence, Set

from sympy import Array, I, Matrix, Symbol
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.qubit import qubit_to_matrix

from ewl.utils import amplitude_to_prob, cache, convert_exp_to_trig, number_of_qubits


class EWL:
    def __init__(self, psi, strategies: Sequence[Matrix], payoff_matrix: Optional[Array] = None):
        assert number_of_qubits(psi) == len(strategies), 'Number of qubits and strategies must be equal'

        if payoff_matrix is not None:
            assert payoff_matrix.rank() == len(strategies) + 1, 'Invalid number of dimensions of payoff matrix'
            assert payoff_matrix.shape == (len(strategies),) + (2,) * len(strategies), 'Invalid shape of payoff matrix'

        self.psi = psi
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix

    @cached_property
    def number_of_players(self) -> int:
        return len(self.strategies)

    @cached_property
    def params(self) -> Set[Symbol]:
        return self.psi.atoms(Symbol).union(*(
            strategy.atoms(Symbol)
            for strategy in self.strategies
        ))

    def fix(self, **kwargs) -> EWL:
        params = {str(symbol): symbol for symbol in self.params}
        replacements = {params[symbol]: value for symbol, value in kwargs.items()}
        psi = self.psi.subs(replacements)
        strategies = [strategy.subs(replacements) for strategy in self.strategies]
        payoff_matrix = self.payoff_matrix.subs(params) if self.payoff_matrix is not None else None
        return EWL(psi, strategies, payoff_matrix)

    @cached_property
    def J(self) -> Matrix:
        C = Matrix([[1, 0], [0, 1]])
        D = Matrix([[0, I], [I, 0]])
        return Matrix.hstack(*[
            TensorProduct(*base) @ qubit_to_matrix(self.psi)
            for base in product((C, D), repeat=number_of_qubits(self.psi))
        ])

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
            assert 0 <= player < self.number_of_players, 'Invalid number of player'

        probs = self.probs(simplify=simplify)
        payoff_matrix = self.payoff_matrix[player] if player is not None else reduce(add, self.payoff_matrix)
        return sum(
            probs[i] * payoff_matrix[idx]
            for i, idx in enumerate(product(range(2), repeat=self.number_of_players))
        )
