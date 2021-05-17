# EWL

A simple Python library to simulate and execute EWL quantum circuits on IBM Q with symbolic calculations using SymPy.

![](https://raw.githubusercontent.com/tomekzaw/ewl/master/docs/ewl.png)

## Installation

```bash
pip install ewl
```

## Examples

-   Prisoner's dilemma
    -   [Two players](https://github.com/tomekzaw/ewl/blob/master/examples/example.ipynb)
    -   [Three players](https://github.com/tomekzaw/ewl/blob/master/examples/three_players.ipynb)
    -   [Payoff function 3D plot](https://github.com/tomekzaw/ewl/blob/master/examples/payoff_function_plot_3d.ipynb)
    -   [Simulation with predefined gate noises](https://github.com/tomekzaw/ewl/blob/master/examples/noise_model.ipynb)

## Usage

```python
from ewl import *

psi = (Qubit('00') + i * Qubit('11')) / sqrt2  # initial state

alice = U(theta=pi / 2, alpha=pi / 2, beta=0)  # quantum strategy
bob = U(theta=0, alpha=0, beta=0)  # classical strategy (C)

payoff_matrix = Array([
    [
        [3, 5],
        [0, 1],
    ],
    [
        [3, 0],
        [5, 1],
    ],
])

ewl = EWL(psi, [alice, bob], payoff_matrix)

ewl.J
ewl.J_H

ewl.amplitudes()
ewl.amplitudes(simplify=False)

ewl.probs()
ewl.probs(simplify=False)

ewl.payoff_function(player=0)
ewl.payoff_function(player=1, simplify=False)
ewl.payoff_function(player=None)

ewl.plot_payoff_function(player=0,
    x=alpha, x_min=0, x_max=pi / 2,
    y=beta, y_min=0, y_max=pi / 2)

ewl.params

ewl_fixed = ewl.fix(alpha=pi / 2, beta=0)

ewl_fixed.draw()

ewl_fixed.draw_transpiled(backend_name='ibmq_athens', optimization_level=3)

ewl_fixed.simulate_probs(backend_name='statevector_simulator')

ewl_fixed.simulate_counts(backend_name='qasm_simulator')

ewl_fixed.run(backend_name='ibmq_athens', optimization_level=3)
```
