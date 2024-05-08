import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the PCNLayer class
class PCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCNLayer, self).__init__()
        self.prediction = nn.Linear(hidden_size, input_size)
        self.error = nn.Linear(input_size, hidden_size)

    def forward(self, input_data, feedback):
        # Compute prediction error
        prediction_error = input_data - feedback
        
        # Update prediction using prediction error
        prediction_update = self.prediction(feedback) + prediction_error
        
        # Update error using prediction error
        error_update = self.error(prediction_error) + input_data
        
        return prediction_update, error_update

# Define the PCN class
class PCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PCN, self).__init__()
        self.layers = nn.ModuleList([PCNLayer(input_size, hidden_size) for _ in range(num_layers)])
        
    def forward(self, input_data):
        feedback = torch.zeros_like(input_data)  # Initialize feedback signal
        
        # Forward pass through PCN layers
        for layer in self.layers:
            prediction, feedback = layer(input_data, feedback)
        
        return prediction

# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define the PCN model
input_size = 28 * 28  # MNIST image size
hidden_size = 128
num_layers = 2
pcn_model = PCN(input_size, hidden_size, num_layers)

# Define the Mean Squared Error (MSE) Loss
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(pcn_model.parameters(), lr=0.01)

# Train the PCN model for reconstruction
for epoch in range(5):  # Train for 5 epochs (you can increase it for better reconstruction)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.view(-1, 28 * 28)  # Flatten the input images
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = pcn_model(inputs)
        
        # Reconstruction loss
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
