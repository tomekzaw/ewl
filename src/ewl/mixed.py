from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, reduce
from operator import add
from typing import Any, List, Optional, Set, Sequence, Tuple

import sympy as sp
from sympy import Array, Matrix, Symbol
from sympy.physics.quantum import TensorProduct

from ewl import BaseEWL
from ewl.utils import cache, number_of_qubits, qubit_to_matrix

Expr = Any


def kraus(U: Matrix, rho: Matrix) -> Matrix:
    return U @ rho @ U.H


@dataclass
class MixedStrategy:
    def __init__(self, strategies: List[Tuple[Expr, Matrix]], *, check_sum: bool = True):
        if check_sum:
            assert sp.simplify(sum(prob for prob, _ in strategies)) == 1, f'Probabilities must sum up to 1'

        self.strategies = strategies

    @cached_property
    def params(self) -> Set[Symbol]:
        return set().union(*(prob.atoms(Symbol) | strategy.atoms(Symbol) for prob, strategy in self.strategies))

    def fix(self, **kwargs) -> MixedStrategy:
        params = {str(symbol): symbol for symbol in self.params}
        replacements = {params[symbol]: value for symbol, value in kwargs.items()}
        return MixedStrategy([
            (prob.subs(replacements), strategy.subs(replacements))
            for prob, strategy in self.strategies
        ])


class MixedEWL(BaseEWL):
    def __init__(self, *, psi, C: Matrix, D: Matrix, players: Sequence[MixedStrategy], payoff_matrix: Optional[Array] = None):
        assert number_of_qubits(psi) == len(players), 'Number of qubits and players must be equal'

        if payoff_matrix is not None:
            assert payoff_matrix.rank() == len(players) + 1, 'Invalid number of dimensions of payoff matrix'
            assert payoff_matrix.shape == (len(players),) + (2,) * len(players), 'Invalid shape of payoff matrix'

        self.psi = psi
        self.C = C
        self.D = D
        self.players = players
        self.payoff_matrix = payoff_matrix

    @cached_property
    def params(self) -> Set[Symbol]:
        return self.psi.atoms(Symbol).union(*(strategy.params for strategy in self.players))

    def fix(self, **kwargs) -> MixedEWL:
        params = {str(symbol): symbol for symbol in self.params}
        replacements = {params[symbol]: value for symbol, value in kwargs.items()}
        psi = self.psi.subs(replacements)
        players = [player.fix(**kwargs) for player in self.players]
        payoff_matrix = self.payoff_matrix.subs(params) if self.payoff_matrix is not None else None
        return MixedEWL(psi=psi, C=self.C, D=self.D, players=players, payoff_matrix=payoff_matrix)

    @cache
    def amplitudes(self) -> Matrix:
        raise NotImplementedError('The state of quantum game with mixed strategies cannot be expressed as a vector, use density matrix instead')

    @cache
    def density_matrix(self) -> Matrix:
        n = self.number_of_players
        I = sp.eye(2)
        psi = qubit_to_matrix(self.psi)
        rho = psi @ psi.H
        for i, mixed in enumerate(self.players):
            rho = reduce(add, (
                prob * kraus(TensorProduct(*[pure if i == j else I for j in range(n)]), rho)
                for prob, pure in mixed.strategies
            ))
        return kraus(self.J.H, rho)

    @cache
    def probs(self, *, simplify: bool = True) -> Matrix:
        probs = self.density_matrix().diagonal()
        if simplify:
            probs = sp.simplify(probs)
        return probs
