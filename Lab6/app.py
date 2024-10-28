import streamlit as st
import torch
import torch.nn.functional as F

# Loss Functions
def cross_entropy_loss(output, target):
    return F.cross_entropy(output.unsqueeze(0), target.unsqueeze(0).long())

def mean_square_error(output, target):
    return torch.mean((output - target) ** 2)

def binary_entropy_loss(output, target):
    return F.binary_cross_entropy(output, target)

# Activation Functions
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum()

def tanh(x):
    return torch.tanh(x)

# Streamlit App
st.title("Loss and Activation Function Calculator")

# Loss Function Inputs
st.subheader("Loss Functions")
input_values = st.text_input("Input tensor (comma-separated)", "0.1, 0.3, 0.6, 0.7")
target_values = st.text_input("Target tensor (comma-separated)", "0.31, 0.32, 0.8, 0.2")

inputs = torch.tensor([float(x) for x in input_values.split(',')])
target = torch.tensor([float(x) for x in target_values.split(',')])

st.write("Mean Square Error:", mean_square_error(inputs, target).item())
st.write("Binary Entropy Loss:", binary_entropy_loss(inputs.sigmoid(), target).item())
st.write("Cross Entropy Loss:", cross_entropy_loss(inputs, target).item())

# Activation Function Inputs
st.subheader("Activation Functions")
activation_values = st.text_input("Activation function input (comma-separated)", "1, 5, -4, 3, -2")

x = torch.tensor([float(val) for val in activation_values.split(',')])

st.write("Sigmoid:", sigmoid(x))
st.write("ReLU:", relu(x))
st.write("Softmax:", softmax(x))
st.write("Tanh:", tanh(x))
