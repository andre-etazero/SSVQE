import numpy as np

from qiskit import QuantumCircuit

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, UnitaryGate

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeBrisbane

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator

import time

from scipy.optimize import minimize

class NotSquareMatrixError(Exception):
    def __init__(self, message = "The hamiltonian and  the initial guess must both be square matrix."):
        self.message = message
        super().__init__(self.message)

class OrderNotMatchError(Exception):
    def __init__(self, message = "The hamiltonian and the initial guess must have the same order."):
        self.message = message
        super().__init__(self.message)

class SSVQE():
    def __init__(self, hamiltonian: np.ndarray, initial_guess: np.ndarray, weighting: list):

        if len(hamiltonian.shape) != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise NotSquareMatrixError


        if hamiltonian.shape != initial_guess.shape:
            raise OrderNotMatchError
        
        self.hamiltonian = hamiltonian
        self.num_states = hamiltonian.shape[0]
        self.initial_guess = initial_guess

        self.num_qubits = int(np.ceil(np.log2(self.num_states)))

        self.weighting = weighting

    def reference_preparation(self, i):
        """
        Prepare the i-th reference state for SSVQE     
        """
        u_gate = UnitaryGate(self.initial_guess)

        reference_circuit = QuantumCircuit(self.num_qubits)

        binary_index = np.binary_repr(i, self.num_qubits)
        for j in range(self.num_qubits):
            if binary_index[-j-1] == '1':
                reference_circuit.x(j)

        reference_circuit.append(u_gate, list(range(self.num_qubits)))
    
        return reference_circuit
    
    def cost_func(slef, params, ansatz_list: list, hamiltonian_list: list, weighting: list, estimator: Estimator, callback_dict: dict):
        """Return callback function that uses Estimator instance,
        and stores intermediate values into a dictionary.

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (Estimator): Estimator primitive instance
            callback_dict (dict): Mutable dict for storing values

        Returns:
            Callable: Callback function object
        """
        pubs = []
        for ansatz, hamiltonian in zip(ansatz_list, hamiltonian_list):
            pubs.append([ansatz, [hamiltonian], [params]])

        job = estimator.run(pubs=pubs)
        result = job.result()

        callback_dict["job_ids"].append(job.job_id())

        cost = 0
        energies = []
        for i in range(len(ansatz_list)):
            energies.append(float(result[i].data.evs[0]))
            cost += weighting[i] * result[i].data.evs[0]
    
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        # Set the prev_vector to the latest one
        callback_dict["prev_params"] = params
        # Compute the value of the cost function at the current vector
        callback_dict["cost_history"].append(cost)
        # Keep trck of the energy expetation values of diferent reference states
        callback_dict["energies_history"].append(energies)
        # Grab the current time
        current_time = time.perf_counter()
        # Find the total time of the execute (after the 1st iteration)
        if callback_dict["iters"] > 1:
            callback_dict["_total_time"] += current_time - callback_dict["_prev_time"]
        # Set the previous time to the current time
        callback_dict["_prev_time"] = current_time
        # Compute the average time per iteration and round it
        time_str = (
            round(callback_dict["_total_time"] / (callback_dict["iters"] - 1), 2)
            if callback_dict["_total_time"]
            else "-"
        )
        # Print to screen on single line

        print(f"Iters. done: {callback_dict['iters']} [Current cost: {cost}]")

        return cost
    
    def execute(self, num_layers=2, initial_parameters_mode='zeros', backend='statevector', method='COBYLA'):

        self.callback_dict = {
            "prev_params": None,
            "iters": 0,
            "job_ids": [],
            "cost_history": [],
            "energies_history": [],
            "_total_time": 0,
            "_prev_time": None,
        }

        self.num_layers = num_layers
        self.method = method

        var_form = RealAmplitudes(num_qubits=self.num_qubits, entanglement='linear', reps=num_layers, insert_barriers=True)
        num_params = var_form.num_parameters

        if initial_parameters_mode == 'zeros':
            self.initial_parameters = np.zeros(num_params)
        elif initial_parameters_mode == 'random':
            self.initial_parameters = 2 * np.pi * np.random.rand(num_params)

        service = QiskitRuntimeService(channel='ibm_quantum', token='ae3a382f16fc6ee05a80eda3e4c7eced6e94a1aa0f9d8e3cceb7fa01304363bc5cd9b8c15151fa294cf43038ee3f19d5a4a5654646402322b601ce36e6a26216')

        if backend == 'statevector':
            self.backend = AerSimulator(method='statevector')
        else:
            self.backend = AerSimulator.from_backend(service.backend(backend))
        
        ansatz_list = []
        for index in range(self.num_states):
            ansatz = self.reference_preparation(index).compose(var_form)
            ansatz_list.append(ansatz)

        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        transpiled_ansatz_list = pm.run(ansatz_list)

        pauli_op = SparsePauliOp.from_operator(self.hamiltonian)

        transplied_op_list = []
        for index in range(self.num_states):
            transplied_op_list.append(pauli_op.apply_layout(transpiled_ansatz_list[index].layout))

        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 10000

            self.result = minimize(
                fun=self.cost_func,
                x0=self.initial_parameters,
                args=(transpiled_ansatz_list, transplied_op_list, self.weighting, estimator, self.callback_dict),
                method=method,
            )

    def get_backend(self):
        try:
            return self.backend
        except:
            print("Backend hasn't been assigned yet, please call the '.execute()' method first.")
            return None
        
    def get_num_layers(self):
        try:
            return self.num_layers
        except:
            print("Number of layers hasn't been assigned yet, please call the '.execute()' method first.")
            return None

    def get_initial_parameters(self):
        try:
            return self.initial_parameters
        except:
            print("Initial parameters hasn't been assigned yet, please call the '.execute()' method first.")
            return None
        
    def get_method(self):
        try:
            return self.method
        except:
            print("Method hasn't been assigned yet, please call the '.execute()' method first.")
            return None
    
    def get_result(self):
        try:
            return self.result
        except:
            print("The job hasn't been created yet, please call the '.execute()' method first.")
            return None
        
    def get_callback_dict(self):
        try:
            return self.callback_dict
        except:
            print("The job hasn't been created yet, please call the '.execute()' method first.")
            return None
    