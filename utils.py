import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        data = self.tensors[0][index]
        target = self.tensors[1][index]
        return data, target, index

    def __len__(self):
        return len(self.tensors[0])

def data_loader(filepath):

    strain_rate_data = np.loadtxt(filepath)

    x = strain_rate_data[:, 0]
    y = strain_rate_data[:, 1]
    z = strain_rate_data[:, 2]

    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    z_tensor = torch.tensor(z, dtype=torch.float32).reshape(-1, 1)

    inputs = torch.cat((x_tensor, y_tensor), dim=1)
    outputs = z_tensor
    return inputs, outputs

def slip_and_scale(inputs, outputs):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    # Initialize scalers for input and output
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # Fit and transform the training data
    x_train_scaled = input_scaler.fit_transform(x_train)
    y_train_scaled = output_scaler.fit_transform(y_train.reshape(-1, 1))  # Reshape if y_train is 1D

    # Transform the test data
    x_test_scaled = input_scaler.transform(x_test)
    y_test_scaled = output_scaler.transform(y_test.reshape(-1, 1))

    # Convert to PyTorch tensors with Float type
    x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    x_test = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_scaled, dtype=torch.float32)

    return x_train, y_train, x_test, y_test, input_scaler, output_scaler


def plot_3D_scatter(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.show()

def extract_max_strain_rate():
    folder_path = ".data/analysis/v1c1"
    strain_rate_data = np.loadtxt(folder_path + "/v1c1/plate_strainrate_data.txt", delimiter=';', skiprows=1)[:,1]
    max_strain_rate  = np.max(strain_rate_data)
    os.rmdir(folder_path)
    return max_strain_rate

def build_inputs(num_inputs, resolution, input_max, input_min):
    # Create min and max arrays based on the number of inputs
    min_values = np.full(num_inputs, input_min)
    max_values = np.full(num_inputs, input_max)

    # Create an array of ranges for each column
    ranges = [np.linspace(min_val, max_val, resolution) for min_val, max_val in zip(min_values, max_values)]

    # Create the meshgrid for all columns
    grids = np.meshgrid(*ranges, indexing='ij')

    # Stack the grids and reshape to create the input matrix
    input_matrix = np.column_stack([grid.ravel() for grid in grids])

    return input_matrix

def load_config(filename):
    with open(filename, 'r') as file:
        content = file.read()
        print("File Content:")
        print(content)  # Check what is being read
        return json.loads(content)  # Load JSON content