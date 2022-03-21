from __future__ import annotations

from functools import cached_property, reduce
from itertools import product
from operator import add
from typing import Optional, Sequence, Set

from sympy import Array, Matrix, Symbol
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.qubit import qubit_to_matrix

from ewl.utils import amplitude_to_prob, cache, convert_exp_to_trig, is_unitary, number_of_qubits


class EWL:
    def __init__(self, *, psi, C: Matrix, D: Matrix, players: Sequence[Matrix], payoff_matrix: Optional[Array] = None):
        assert number_of_qubits(psi) == len(players), 'Number of qubits and players must be equal'

        for i, player in enumerate(players):
            assert is_unitary(player), f'Player {i} strategy is not unitary'

        if payoff_matrix is not None:
            assert payoff_matrix.rank() == len(players) + 1, 'Invalid number of dimensions of payoff matrix'
            assert payoff_matrix.shape == (len(players),) + (2,) * len(players), 'Invalid shape of payoff matrix'

        self.psi = psi
        self.C = C
        self.D = D
        self.players = players
        self.payoff_matrix = payoff_matrix

    @cached_property
    def number_of_players(self) -> int:
        return len(self.players)

    @cached_property
    def params(self) -> Set[Symbol]:
        return self.psi.atoms(Symbol).union(*(
            player.atoms(Symbol)
            for player in self.players
        ))

    def fix(self, **kwargs) -> EWL:
        params = {str(symbol): symbol for symbol in self.params}
        replacements = {params[symbol]: value for symbol, value in kwargs.items()}
        psi = self.psi.subs(replacements)
        players = [player.subs(replacements) for player in self.players]
        payoff_matrix = self.payoff_matrix.subs(params) if self.payoff_matrix is not None else None
        return EWL(psi=psi, C=self.C, D=self.D, players=players, payoff_matrix=payoff_matrix)

    @cached_property
    def J(self) -> Matrix:
        return Matrix.hstack(*[
            TensorProduct(*base) @ qubit_to_matrix(self.psi)
            for base in product((self.C, self.D), repeat=number_of_qubits(self.psi))
        ])

    @cached_property
    def J_H(self) -> Matrix:
        return self.J.H

    @cache
    def amplitudes(self, *, simplify: bool = True) -> Matrix:
        ampl = self.J_H @ TensorProduct(*self.players) @ qubit_to_matrix(self.psi)
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
