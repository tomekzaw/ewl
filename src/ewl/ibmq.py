from functools import cached_property
from typing import Dict, Optional

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.extensions.unitary import UnitaryGate
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.ibmq.exceptions import IBMQProviderError
from qiskit.quantum_info.operators import Operator
from qiskit.tools import job_monitor

from ewl import EWL
from ewl.utils import sympy_to_numpy_matrix


class EWL_IBMQ:
    def __init__(self, ewl: EWL, *, noise_model: Optional[NoiseModel] = None):
        if ewl.params:
            raise Exception('Please provide values for the following parameters: ' + ', '.join(map(str, ewl.params)))

        self.ewl = ewl
        self.noise_model = noise_model

    @cached_property
    def _provider(self) -> AccountProvider:
        try:
            return IBMQ.get_provider()
        except IBMQProviderError:
            raise RuntimeError('Please run this notebook on https://quantum-computing.ibm.com/lab '
                               'or save account token using IBMQ.save_account function')

    def _make_qc(self, *, measure: bool) -> QuantumCircuit:
        J = UnitaryGate(Operator(sympy_to_numpy_matrix(self.ewl.J)), label='$J$')
        J_H = UnitaryGate(Operator(sympy_to_numpy_matrix(self.ewl.J_H)), label='$J^\\dagger$')

        all_qbits = range(self.ewl.number_of_players)

        qc = QuantumCircuit(self.ewl.number_of_players)
        qc.append(J, all_qbits)
        qc.barrier()

        for i, player in enumerate(self.ewl.players):
            U_i = UnitaryGate(Operator(sympy_to_numpy_matrix(player)), label=f'$U_{{{i}}}$')
            qc.append(U_i, [i])

        qc.barrier()
        qc.append(J_H, all_qbits)

        if measure:
            qc.measure_all()

        return qc

    @cached_property
    def qc(self) -> QuantumCircuit:
        return self._make_qc(measure=True)

    def draw(self):
        return self.qc.draw('mpl')

    def draw_transpiled(self, backend_name: str, *, optimization_level: int = 3):
        backend = self._provider.get_backend(backend_name)
        transpiled_qc = transpile(self.qc, backend, optimization_level=optimization_level)
        return transpiled_qc.draw('mpl')

    def simulate_probs(self, backend_name: str = 'statevector_simulator') -> Dict[str, float]:
        circ = self._make_qc(measure=False)
        simulator = Aer.get_backend(backend_name)
        return execute(circ, simulator).result().get_counts()

    def simulate_counts(self) -> Dict[str, int]:
        simulator = AerSimulator(noise_model=self.noise_model)
        circ = transpile(self.qc, simulator) if self.noise_model is not None else self.qc
        result = simulator.run(circ).result()
        return result.get_counts()

    def run(self, backend_name: str = 'least_busy', *, optimization_level: int = 3) -> Dict[str, int]:
        if backend_name == 'least_busy':
            small_devices = self._provider.backends(
                filters=lambda x: x.configuration().n_qubits >= self.ewl.number_of_players
                                  and not x.configuration().simulator and x.status().operational)  # noqa: W503, E131
            backend = least_busy(small_devices)
        else:
            backend = self._provider.get_backend(backend_name)

        job = execute(self.qc, backend, optimization_level=optimization_level)
        job_monitor(job)
        return job.result().get_counts()
