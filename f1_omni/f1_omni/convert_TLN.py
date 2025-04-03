import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

class TLN_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, 10)
        self.conv2 = nn.Conv1d(24, 36, 8)
        self.conv3 = nn.Conv1d(36, 48, 4)
        self.conv4 = nn.Conv1d(48, 64, 3)
        self.conv5 = nn.Conv1d(64, 64, 3)
        self.fc1 = nn.Linear(704, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

tflite_model_path = "/home/r478a194/Downloads/f1_tenth_model_small_noquantized.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()
tln_pytorch = TLN_PyTorch()
tln_pytorch_dict = {}

for tensor_info in tensor_details:
    name = tensor_info['name']
    tensor = interpreter.get_tensor(tensor_info['index'])

    if "Conv1D" in name and "BiasAdd" not in name:
        tensor = np.transpose(tensor, (0, 2, 1))
    elif "MatMul" in name:
        tensor = np.transpose(tensor)

    tensor = torch.tensor(tensor)

    param_map = {
        "conv1/Conv1D": "conv1.weight",
        "conv1/BiasAdd": "conv1.bias",
        "conv2/Conv1D": "conv2.weight",
        "conv2/BiasAdd": "conv2.bias",
        "conv3/Conv1D": "conv3.weight",
        "conv3/BiasAdd": "conv3.bias",
        "conv4/Conv1D": "conv4.weight",
        "conv4/BiasAdd": "conv4.bias",
        "conv5/Conv1D": "conv5.weight",
        "conv5/BiasAdd": "conv5.bias",
        "dense/MatMul": "fc1.weight",
        "dense/BiasAdd": "fc1.bias",
        "dense_1/MatMul": "fc2.weight",
        "dense_1/BiasAdd": "fc2.bias",
        "dense_2/MatMul": "fc3.weight",
        "dense_2/BiasAdd": "fc3.bias",
        "dense_3/MatMul": "fc4.weight",
        "dense_3/BiasAdd": "fc4.bias",
    }

    for key, value in param_map.items():
        if key in name:
            tln_pytorch_dict[value] = tensor

tln_pytorch.load_state_dict(tln_pytorch_dict)
