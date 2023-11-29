"""
SNN Model for Shenjing Chip

Version 1.3
- Change the number of epochs to avoid over-fitting.
- Change the information to print
- Change the LIF model: 
    - no decay
    - modify >= Threshold
    - no bias


Andres Brito
2023 - 11- 13

Dataset: MNIST  

Frameworks:  
snnTorch (PyTorch)  --> Network  
Bindsnet --> Dataset and encoding  

Network Architecture: 1 hidden layer + 1 output layer

- Datasets are previously generated using Bindsnet --> Encoding using Poisson Distribution
"""

###########################################################################################
# Libraries
###########################################################################################
import snntorch as snn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import pickle
import os


###########################################################################################
# Global Variables
###########################################################################################
# Seed for reproducibility
torch.manual_seed(123)

# DataLoader arguments
batch_size = 128
dtype = torch.float

# Set Device to run: CPU or GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Running on {device}\n')


###########################################################################################
# Prepare Train and Test Datasets
###########################################################################################
# Retrieve the information from Bindsnet (Poisson Distribution) results: Training
with open('./data/mnist_train.pkl', 'rb') as f:
    # Deserialize the file
    mnist_train = pickle.load(f)
print(f'Number of samples in the training dataset: {len(mnist_train)}')

# Retrieve the information from Bindsnet (Poisson Distribution) results: Testing
with open('./data/mnist_test.pkl', 'rb') as f:
    # Deserialize the file
    mnist_test = pickle.load(f)
print(f'Number of samples in the testing dataset: {len(mnist_test)}\n')

# Create DataLoaders
# Drop last --> refers to drop the last bacth if it is incomplete
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


###########################################################################################
# Prepare the Model
###########################################################################################
# Network Architecture
# Considering the MNIST MLP from Shenjing Paper
num_inputs = 28*28
num_hidden = 512
num_outputs = 10

# Temporal Dynamics
num_steps = 20

# Represents the decay of the membrane potential
# No decay
beta = 1

# After spiking, the membrane potential is reduced by threshold value

# The firing function has been modified to >= Threshold

# No bias

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(1)):           # data.size(1) =  number of time steps
            cur1 = self.fc1(x[:,step,:])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)
print(f'SNN Architecture: \n {net} \n')

###########################################################################################
# Train the Model
###########################################################################################
# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, 20, -1))
    # print(output.shape)
    _, idx = output.sum(dim=0).max(1)
    # print(idx.shape)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Batch number {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

# Loss definition
loss = nn.CrossEntropyLoss()

# Optimizer --> this learning rate performs well for RNN
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 20
loss_hist = []
loss_epoch = []
test_loss_hist = []
test_loss_epoch = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data_full in train_batch:
        # Get the encoded data
        data = data_full['encoded_image']
        data = data.float()    
 
        targets = data_full['encoded_label']
        
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, 20, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_d_full = next(iter(test_loader))

            # Get the encoded data
            test_data = test_d_full['encoded_image']
            test_data = test_data.float()    
 
            test_targets = test_d_full['encoded_label']
             
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, 20, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy several times per 5 epochs
            if iter_counter % 100 == 0 and epoch % 5 == 0:
                train_printer()
            counter += 1
            iter_counter +=1
    
    # Average loss per epoch
    # print(epoch*iter_counter)
    loss_epoch.append(sum(loss_hist[epoch*iter_counter:]) / iter_counter)
    test_loss_epoch.append(sum(test_loss_hist[epoch*iter_counter:]) / iter_counter)

# Loss per epoch
# print(f'Training Loss per epoch: {loss_epoch}')
# print(f'Testing Loss per epoch: {test_loss_epoch}')

# Create the folder
os.makedirs('./results', exist_ok = True) 

# Plot Loss per Batch
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves per Batch")
plt.legend(["Train Loss per Batch", "Test Loss per Batch"])
plt.xlabel("Batches")
plt.ylabel("Loss")
# Save the image
plt.savefig('./results/loss_batch_3.png')

# Plot Loss per Epoch
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_epoch)
plt.plot(test_loss_epoch)
plt.title("Loss Curves per Epoch")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
# Save the image
plt.savefig('./results/loss_epoch_3.png')

###########################################################################################
# Test the Model
###########################################################################################
# Testing loop
total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  net.eval()
  for data_full in test_loader:
    data = data_full['encoded_image']
    data = data.float()    
 
    targets = data_full['encoded_label']
    
    data = data.to(device)
    targets = targets.to(device)

    # forward pass
    test_spk, _ = net(data.view(data.size(0), 20, -1))

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}% \n")

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# Save the model for inference
torch.save(net.state_dict(), './results/snnMNIST_model1_3.pt')