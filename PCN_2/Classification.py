import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


losses = [] #list for visualizing loss

#defining all necessary methods here, maybe I need to put them into PCNLayer class, unsure yet



# Define the PCNLayer class, maybe I only need to have self.state as a tensor and not the other variables
class PCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCNLayer, self).__init__()
        self.prediction = nn.Linear(hidden_size, input_size)
        self.error = nn.Linear(input_size, hidden_size)
        self.state = torch.rand(hidden_size) #state is just the tensor that is saved in the layer
        self.hidden_size = hidden_size
    
    
    
    def predict(self):
        random_matrix = torch.ones((784, 784))
        prediction = torch.matmul(self.state, random_matrix)
        return prediction

    def error_calculation(self, prediction):
        error = self.state - prediction
        return error

    def recalculate_state(self,error):
        state = self.state + error #placeholder
        print(state)
        return state


    
    
# Define the PCN class
class PCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(PCN, self).__init__()
        self.layers = nn.ModuleList([PCNLayer(input_size, hidden_size) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.classifier.requires_grad_(False) # Prevent the layer from being updated during training
        self.relu = nn.ReLU() #ReLu layers which can be added
        self.bottom_layer = torch.zeros(hidden_size) #placeholder for now

        
        
    def forward(self, input_data):
        cycles = 10  #Number is subject to change
        #feedback = self.relu(feedback) #add ReLu layer for non-linearity

        # Forward pass through PCN layers
        for _ in range(cycles): #first try at getting it cyclic, right now it just repeats multiple times
            predictions = []
            errors = []
            

            #First all the predictions are calculated, then the errors and then the states are recalculated, so I don't have to access multiple layers at once
            for layer in self.layers:
                prediction = layer.predict()
                predictions.append(prediction)
            for i in range(len(self.layers)):
                errors.append(self.layers[i].error_calculation(predictions[i]))
            for i in range(len(self.layers)):
                if i > 0: #First layer is fixed
                    self.layers[i].state = self.layers[i].recalculate_state(errors[i])
            
        # Classification layer
       
        bottom_layer = input_data #placeholder ish atm
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
input_size = 784  # MNIST image size
hidden_size = 784
num_layers = 10
pcn_model = PCN(input_size, hidden_size, num_layers)
pcn_model.forward(trainloader)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pcn_model.parameters(), lr=0.01)

# Train the PCN model
'''for epoch in range(5):  # Train for 5 epochs (you can increase it for better performance)
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
        losses.append(running_loss / len(trainloader))'''

# Visualize the training loss

print('Finished Training')