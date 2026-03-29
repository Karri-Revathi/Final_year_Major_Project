import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

# Quantum Setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# MUST match your saved weights
weight_shapes = {"weights": (2, n_qubits)}


# Quantum Circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    inputs = inputs[0]   # take first sample from batch

    # Encode data
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Trainable quantum layers
    for i in range(n_qubits):
        qml.RZ(weights[0][i], wires=i)
        qml.RX(weights[1][i], wires=i)

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# Convert to Torch layer
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet18
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, n_qubits)

        # Quantum layer
        self.quantum = qlayer

        # Final classifier
        self.fc = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.cnn(x)        # [batch, 4]
        x = self.quantum(x)    # [4]
        x = self.fc(x)         # [2]
        return x