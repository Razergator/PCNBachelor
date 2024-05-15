import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


losses = [] #list for visualizing loss

# Define the PCNLayer class
class PCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCNLayer, self).__init__()
        self.prediction = nn.Linear(hidden_size, input_size)
        self.error = nn.Linear(input_size, hidden_size)
        self.state = torch.random(hidden_size) #state is just the tensor that is saved in the layer
        self.hidden_size = hidden_size

    def forward(self, input_data, feedback):
        # Initialize feedback with zeros
        feedback = torch.zeros(self.hidden_size)
        # Forward pass (error computation)
        prediction_error = input_data - feedback
        # Update feedback using error
        feedback = self.error(prediction_error)
        
        return feedback, prediction_error
    
    def backward(self, feedback):
        # Initialize prediction with zeros
        prediction = torch.zeros(self.hidden_size)
        # Backward pass (prediction computation)   
        prediction = self.prediction(feedback)

        return prediction
    
    
# Define the PCN class
class PCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(PCN, self).__init__()
        self.layers = nn.ModuleList([PCNLayer(input_size, hidden_size) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.classifier.requires_grad_(False) # Prevent the layer from being updated during training
        self.relu = nn.ReLU() #ReLu layers which can be added
        
    def forward(self, input_data):
        feedback = torch.zeros_like(input_data)  # Initialize feedback signal
        
        # Forward pass through PCN layers
        for layer in self.layers:
            prediction, feedback = layer(input_data, feedback)
            feedback = self.relu(feedback) #add ReLu layer for non-linearity
            
        # Backward pass through PCN layers
        for layer in list(reversed(self.layers)):
            prediction = layer.backward(prediction)
            
            return prediction
        
        # Classification layer
        bottom_layer = self.classifier(prediction) #placeholder ish atm
        return bottom_layer

   

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
hidden_size = 784
num_layers = 2
pcn_model = PCN(input_size, hidden_size, num_layers)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pcn_model.parameters(), lr=0.01)

# Train the PCN model
for epoch in range(5):  # Train for 5 epochs (you can increase it for better performance)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 28 * 28)  # Flatten the input images
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = pcn_model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        losses.append(running_loss / len(trainloader))


def my_prediction_method(self, input_data):
        # Define your own prediction method here
        # For example, you can use a linear transformation followed by a softmax activation
        weight = torch.nn.Parameter(torch.zeros())
        bias = torch.nn.Parameter(torch.zeros())
        output = F.linear(input_data, weight, bias)
        output = F.softmax(output, dim=1)
        return output

def error_calculation(self, layer_state_1, prediction):
        # Define your own error calculation method here
        # For example, you can use the cross-entropy loss
        error = layer_state_1 - prediction
        return error



# Visualize the training loss
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print('Finished Training')