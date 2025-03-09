
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv0 = nn.Conv2d(1, 24, (3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(24, 128)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)
        self.act4 = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv0(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        return x

# Initialize the model
model = NeuralNetwork()

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        x_batch, y_batch = batch
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "trained_model.pth")


