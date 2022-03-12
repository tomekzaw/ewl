from math import log2

import numpy as np
import sympy as sp
from sympy import Abs, init_printing, Matrix, sin  # noqa: F401
from sympy.physics.quantum.qubit import qubit_to_matrix

try:
    from functools import cache  # Python 3.9+
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)


def number_of_qubits(psi) -> int:
    return int(log2(len(qubit_to_matrix(psi))))


def convert_exp_to_trig(expr):
    return expr.rewrite(sp.sin).simplify()


def amplitude_to_prob(expr):
    return sp.Abs(expr) ** 2


def sympy_to_numpy_matrix(matrix: Matrix) -> np.array:
    return np.array(matrix).astype(complex)
