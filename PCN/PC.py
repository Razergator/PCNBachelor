import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network model
class PCNet(nn.Module):
    def __init__(self):
        super(PCNet, self).__init__()
        self.fc = nn.Linear(1, 1)  # One input feature, one output feature

    def forward(self, x):
        return self.fc(x)

# Sample data: input and output
X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)

# Initialize the neural network model
model = PCNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):  # 1000 epochs for demonstration
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.item()))

# Make predictions
predictions = model(X)

# Print the predictions
print("Predictions:", predictions.detach().numpy())
