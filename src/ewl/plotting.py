from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ewl import EWL


def ignore_matplotlib_warnings():
    from warnings import simplefilter
    from matplotlib import MatplotlibDeprecationWarning

    simplefilter('ignore', category=MatplotlibDeprecationWarning)


def plot_payoff_function(ewl: EWL, *, player: Optional[int],
                         x: sp.Symbol, x_min, x_max, y: sp.Symbol, y_min, y_max, n: int = 20,
                         figsize=(8, 8), cmap=plt.cm.coolwarm, **kwargs) -> plt.Figure:
    f = sp.lambdify([x, y], ewl.payoff_function(player=player))

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
