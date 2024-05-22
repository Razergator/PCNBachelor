import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


losses = [] #list for visualizing loss

#defining all necessary methods here, maybe I need to put them into PCNLayer class, unsure yet



# Define the PCNLayer class, maybe I only need to have self.state as a tensor and not the other variables
class PCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCNLayer, self).__init__()
        self.state = torch.rand(hidden_size,requires_grad=True) #state is just the tensor that is saved in the layer
        self.hidden_size = hidden_size
    
    
    
    def predict(self):
        W_matrix = torch.ones((10, 10), requires_grad=True) #placeholder for now
        prediction = torch.matmul(self.state, W_matrix)
        
        return prediction

    def error_calculation(self, prediction):
        error = self.state - prediction
        
        return error

    def recalculate_state(self,error, learning_rate = 0.01):
        self.state = self.state - error
        print(self.state)

        
        return self.state


    
    
# Define the PCN class
class PCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(PCN, self).__init__()
        self.layers = nn.ModuleList([PCNLayer(input_size, hidden_size) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() #ReLu layers which can be added
        self.bottom_layer = torch.zeros(hidden_size) #placeholder for now

        self.layers[num_layers-1].state = torch.tensor([0,1,2,3,4,5,6,7,8,9]).float().requires_grad_(True) #placeholder for now
        
        
        
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
                if i > 0 and i < len(self.layers)-1:  #First layer and last layers are fixed
                    self.layers[i].state = self.layers[i].recalculate_state(errors[i])
        
        out = self.classifier(self.layers[num_layers-1].state) #output of the last layer is the input for the classifier

        return out
            
                   

           
            
        # Classification layer
       
       
        

   

# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True) #is normally 64, i dont really get it to work now

# Define the PCN model
input_size = 24*24  # MNIST image size
hidden_size = 10
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
        labels = labels.float()
        
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