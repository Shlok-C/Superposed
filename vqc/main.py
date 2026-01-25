# pennylane qml imports
import pennylane as qml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

# various qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, SparsePauliOp, random_clifford

# qiskit ml imports
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from scipy.stats import unitary_group

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from tqdm import trange

classes = {
    'separable': [],
    'linear': [(0, 1), (1, 2)],
    'biseparable-1': [(0, 1)],
    'biseparable-2': [(1, 2)],
    'biseparable-3': [(0, 2)],
    'star': [(0, 1), (0, 2), (1, 2)]
}

def get_graph_state(graph: dict, n_qubits: int=3) -> Statevector:
    '''
    Function for deciphering quantum state from a graph object
    
    :param graph: graph relations (verticies and edges)
    :type graph: dict
    :param n_qubits: Num qubits
    :type n_qubits: int
    :return: Returns Statevector of the graph state
    :rtype: Statevector
    '''
    circuit = QuantumCircuit(3)

    for i in range(n_qubits):
        circuit.h(i)

    for i, j in graph:
        circuit.cz(i, j)

    state = Statevector.from_instruction(circuit)

    return state
  

def apply_random_clifford(state: Statevector, n_qubits: int=3) -> Statevector:
    '''
    Apply random LC group matrix and return statevector
    
    :param state: State to be transformed
    :type state: Statevector
    :param n_qubits: Num qubits
    :return: Transformed statevector
    :rtype: Statevector
    '''

    circuit = QuantumCircuit(n_qubits)

    # apply random clifford gates 
    for i in range(n_qubits):
        cliff_gate = random_clifford(1)
        circuit.compose(cliff_gate, qubits=[i], inplace=True)

    post_LC_state = state.evolve(circuit)

    return post_LC_state

def apply_random_unitary(state: Statevector, n_qubits: int=3) -> Statevector:
    '''
    Apply random LU group matrix and return statevector
    
    :param state: State to be transformed
    :type state: Statevector
    :param n_qubits: Num qubits
    :return: Transformed statevector
    :rtype: Statevector
    '''

    circuit = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        U_mat = unitary_group.rvs(2)

        # get random unitary from unitary group, convert to qiskit gate
        U = UnitaryGate(U_mat)

        circuit.compose(U, qubits=[i], inplace=True)

    post_LU_state = state.evolve(circuit)
    return post_LU_state

def generate_data_set(label: str, entanglement_classes=classes, target_class: str='linear', samples: int=1000, use_LC: bool=False, shuffle=True) -> tuple[list, list]:
    '''
    Generate data set of random graph states for a certain target entanglement classification
    
    :param target_class: entanglement classification
    :type target_class: str
    :param samples: Num samples for total dataset
    :type samples: int
    :param use_LC: Flag to use Local Unitary or Local Clifford transformations
    :type use_LC: bool
    :return: Return tuple containing list of feature vectors and labels
    :rtype: tuple[list, list]
    '''

    X = []
    y = []

    DESC_WIDTH = max(
        len(f"Added random LU equivalent {name} states to {label} data") for name in entanglement_classes
    )

    target_samples = int(samples * 0.5)
    other_samples = int(samples * 0.5)

    # generate half of the samples in the target orbit (linear in default case) with label '-1'
    for _ in trange(target_samples, desc=f"Added random {'LC' if use_LC else 'LU'} equivalent {target_class} states to {label} data".ljust(DESC_WIDTH)):
        base_state = get_graph_state(entanglement_classes[target_class])
        
        transformed_state = apply_random_unitary(base_state) if not use_LC else apply_random_clifford(base_state)

        transformed_state.draw(output='text')

        X.append(transformed_state)
        y.append(-1)

    samples_per_other_class = other_samples // (len(entanglement_classes) - 1)
    for name, cl in entanglement_classes.items():
        if cl == entanglement_classes[target_class]: continue

        # 100 for each of the other entanglement classes (500 total)
        for _ in trange(samples_per_other_class, desc=f"Added random {'LC' if use_LC else 'LU'} equivalent {name} states to {label} data".ljust(DESC_WIDTH)):
            base_state = get_graph_state(cl)

            transformed_state = apply_random_unitary(base_state) if not use_LC else apply_random_clifford(base_state)

            X.append(transformed_state)
            y.append(1)

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )

    # shuffle
    indices = torch.randperm(len(X))
    X_shuffled = [X[i] for i in indices]
    y_shuffled = y_tensor[indices]

    return X_shuffled, y_shuffled

# X_train, y_train = generate_data_set('training', target_class='linear', samples=800, use_LC=True)
# X_test, y_test = generate_data_set('testing', target_class='linear', samples=200, use_LC=True)

class VariationalCircuit:

    def __init__(self, n_qubits: int=3, n_layers: int=2):
        '''
        Build variational circuit for hybrid quantum/classical entanglement classification

        1. Amplitude Encoding
        2. Variational Parameter rotations
        3. Entanglement ladder (CNOTs)
        4. Measurement
        
        :param n_qubits: Num qubits
        '''
        self.n_qubits = n_qubits
        self.n_layers = n_layers
    
        self.circuit, self.paramaters = self.build_circuit()

    def build_circuit(self):

        params_per_qubit = 3
        num_params = params_per_qubit * self.n_qubits * self.n_layers

        # parameters for rotation gates
        circuit = QuantumCircuit(self.n_qubits)

        theta = ParameterVector('θ', num_params)

        parameter_index = 0
        for _ in range(self.n_layers):

            for qubit in range(self.n_qubits):
                circuit.rx(theta[parameter_index], qubit)
                circuit.ry(theta[parameter_index + 1], qubit)
                circuit.rz(theta[parameter_index + 2], qubit)
                parameter_index += 3

            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit+1)

            circuit.cx(self.n_qubits-1, 0)

            circuit.barrier()

        return circuit, theta
    
    def evolve_state(self, input_state: Statevector, param_values: list):
        # vary parameters to circuit
        bound_circuit = self.circuit.assign_parameters(param_values)

        # Evolve
        final_state = input_state.evolve(bound_circuit)

        # Measure for some observable

        # first build measurement operator
        I = np.identity(2)
        Z = np.array([
            [1, 0],
            [0, -1]
        ])

        Z_operators = []

        for target in range(self.n_qubits):
            ops = [I] * self.n_qubits
            ops[self.n_qubits - 1 - target] = Z

            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)

            Z_operators.append(result)

        expectation_values = []
        psi = final_state.data
        # Calculate expectation values through ⟨ψ|Z|ψ⟩

        for operator in Z_operators:
            ev = np.real(np.conj(psi) @ operator @ psi)
            expectation_values.append(ev)

        return expectation_values



    def __repr__(self):
        print(self.circuit)
        return ''

# qc = VariationalCircuit()

# print(qc)

# sample_input = X[7]
# random_params = np.random.randn(18) * 0.1

# output = qc.evolve_state(sample_input, random_params)
# print(f"\nVQC output (3 expectation values): {output}")
# print(f"Output range: [{np.array(output).min():.3f}, {np.array(output).max():.3f}]")

class QuantumNeuralNetwork(nn.Module):

    def __init__(self, n_qubits: int=3, n_layers: int=2, hidden_dims: list[int]=[16, 8]):
        super(QuantumNeuralNetwork, self).__init__()
        
        self.vqc: VariationalCircuit = VariationalCircuit(n_qubits, n_layers)

        # set random initial parameters
        # Params -> θ = {α, β, γ}
        num_params = len(self.vqc.paramaters)
        self.quantum_parameters = nn.Parameter(
            torch.randn(num_params) * 0.1
        )

        layers = []

        # input layer -> 'encode' observable measurements (⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩) -> first hidden layer
        layers.append(nn.Linear(n_qubits, hidden_dims[0]))
        layers.append(nn.Tanh())

        # hidden layers
        for layer in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer+1]))
            layers.append(nn.Tanh())

        # output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)


    # def prepare_datasets(X_train, y_train, X_test, y_test, batch_size=16):
    #     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    #     y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    #     train_indices = list(range(len(X_train)))
    #     test_indices = list(range(len(X_test)))
        
    #     return (X_train, y_train_tensor), (X_test, y_test_tensor)

    def forward_pass(self, input_states: list):

        # batch_size = len(input_states)
        quantum_outputs = []
        
        # calculate expectation values for every state (forward pass through quantum circuit)
        for state in input_states:

            exp_vals = self.vqc.evolve_state(
                state,
                self.quantum_parameters.detach().numpy() # returns as np array (casting from torch tensor)
            )

            quantum_outputs.append(exp_vals)

        quantum_features = torch.tensor(
            quantum_outputs,
            dtype=torch.float32
        )

        preds = self.network(quantum_features)
        return preds
    
    def train_model(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=16, learning_rate=0.1):
        
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # training history
        training_losses = []
        training_accuracies = []
        testing_accuracies = []

        num_batches = len(X_train) // batch_size

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            correct_train = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                predictions = self.forward_pass(X_batch)
                
                # calculate loss
                loss = criterion(predictions, y_batch.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                
                # track metrics
                epoch_loss += loss.item()
                pred_labels = torch.sign(predictions)
                correct_train += (pred_labels == y_batch.unsqueeze(1)).sum().item()
            
            # accuracies
            avg_loss = epoch_loss / num_batches
            train_acc = 100.0 * correct_train / len(X_train)
            
            # eval w/ test set
            self.eval()
            with torch.no_grad():
                test_predictions = self.forward_pass(X_test)
                test_pred_labels = torch.sign(test_predictions)
                correct_test = (test_pred_labels == y_test.unsqueeze(1)).sum().item()
                test_acc = 100.0 * correct_test / len(X_test)
            
            training_losses.append(avg_loss)
            training_accuracies.append(train_acc)
            testing_accuracies.append(test_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1:3d}/{epochs} ] "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:6.2f}% | "
                    f"Test Acc: {test_acc:6.2f}%"
                )
        
        print(f"\nFinal Training Accuracy: {training_accuracies[-1]:.2f}%")
        print(f"Final Test Accuracy:     {testing_accuracies[-1]:.2f}%")
        
        return training_losses, training_accuracies, testing_accuracies

    # Visualization function with types
    def plot_training_results(
        self,
        train_losses,
        train_accs,
        test_accs, 
        target_class: str
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax1.plot(train_losses, linewidth=2, color='#2E86AB')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MSE Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(train_accs, label='Training', linewidth=2, color='#2E86AB')
        ax2.plot(test_accs, label='Test', linewidth=2, color='#A23B72')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'Accuracy: {target_class.upper()} Class', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.show()

    # Training pipeline with types
    def train_for_class(
        self,
        target_class: str,
        entanglement_classes: dict,
        n_train: int = 800,
        n_test: int = 200,
        use_LC: bool = False,
        epochs: int = 50,
        learning_rate: float = 0.01
    ):
        print(f"Training VQC for class: {target_class.upper()}")
        print(f"Out of {entanglement_classes}")
        print(f"Classification type: {'LC orbit' if use_LC else 'LU orbit'}")
        
        # Generate data
        print("\n[1/3] Generating training data...")
        X_train, y_train = generate_data_set(
            label='training',
            entanglement_classes=entanglement_classes,
            target_class=target_class,
            samples=n_train,
            use_LC=use_LC
        )
        
        print("\n[2/3] Generating test data...")
        X_test, y_test = generate_data_set(
            label='testing',
            entanglement_classes=entanglement_classes,
            target_class=target_class,
            samples=n_test,
            use_LC=use_LC
        )
        
        print(f"\nDataset ready:")
        print(f"Training: {len(X_train)} samples")
        print(f"Test: {len(X_test)} samples")
        print(f"Class balance: {(y_train == -1).sum().item():.0f} target / {(y_train == 1).sum().item():.0f} other")
        
        train_losses, train_accs, test_accs = self.train_model(
            X_train, X_test, y_train, y_test,
            epochs=epochs,
            batch_size=16,
            learning_rate=learning_rate
        )
        
        # Plot results
        self.plot_training_results(train_losses, train_accs, test_accs, target_class)
        
        return (train_losses, train_accs, test_accs)

# Run training
if __name__ == "__main__":
    model = QuantumNeuralNetwork()
    print(model)
    print(model.vqc)

    metrics = model.train_for_class(
        target_class='linear',
        n_train=900,
        n_test=100,
        use_LC=False,  # LC orbit classification
        epochs=50,
        learning_rate=0.05
    )


            