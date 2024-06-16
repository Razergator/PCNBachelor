import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


losses = [] #list for visualizing loss

W = torch.empty(64, 64) #duplikat bei recalculate state einfügen mit richtigen dimensionen
torch.nn.init.xavier_uniform_(W)

def relu_prime(x): #ReLu derivative is 1 if x > 0, 0 otherwise
    return (x > 0).float()

#own optimizer will be defined here
def matrix_update(learning_rate, error, state_1):#
    #return learning_rate * (error * relu_prime(state_1).T)
    #pass
    #print("error shape:", error.shape)
    #print("relu_prime(state_1).T shape:", relu_prime(state_1).T.shape)
    new_W = learning_rate * (error * relu_prime(state_1).T)
    print("test",new_W.shape)
    W = new_W 
    return new_W
    

# Define the PCNLayer class, maybe I only need to have self.state as a tensor and not the other variables
class PCNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PCNLayer, self).__init__()
        self.output = output_size
        self.linear = nn.Linear(input_size, output_size, bias=False)#torch.ones((10, 10), requires_grad=True)
        self.relu = nn.ReLU()
    
    
    
    def predict(self):
        relu_state = self.relu(self.state)
        prediction = self.linear(relu_state)
        return prediction

    def error_calculation(self, prediction):
        #print("prediction:", prediction.shape)
        #print("State:", self.state.shape)
        error = self.state - prediction
        
        return error

    def recalculate_state(self, learning_rate, e_0,e_1, W): #e_i = errsor, W = weight_matrix, phi_prime = derivative of an activation function(here ReLu)
        phi_prime = relu_prime(self.state)
        #print("phi_prime",phi_prime.shape)
        #print("W.T@e_1",(W.T @ e_1).shape)
        
        self.state += learning_rate * (-e_0 + (phi_prime))  * (W.T @ e_1)
        print(self.state) 
       


    
    
# Define the PCN class
class PCN(nn.Module):
    def __init__(self, hidden_sizes, W):
        super(PCN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.W = W
        
        
        #print("W",W)
        for i in range(len(hidden_sizes)):
            if i == 0:
                print("i=0",hidden_sizes[i])
                self.layers.append(PCNLayer(hidden_sizes[i],1))
                print("layer:",self.layers[i])
            if i > 0:
                print("i>0",hidden_sizes[i],hidden_sizes[i-1])
                self.layers.append(PCNLayer(hidden_sizes[i],hidden_sizes[i-1]))
                print("layer:",self.layers[i])
        

    def forward(self, input_data):
        cycles = 10  #Number is subject to change
        
        # Forward pass through PCN layers
        for i in range(len(self.layers)):
            #if i > 0:
            self.layers[i].state = torch.rand(64,self.hidden_sizes[i])
            print("sizes", self.layers[i].state.shape) 

        for _ in range(cycles): 
            predictions = []
            errors = []
            
            #First all the predictions are calculated, then the errors and then the states are recalculated, so I don't have to access multiple layers at once
            for layer in self.layers[1:]: #first layer no prediciton
                prediction = layer.predict()
                predictions.append(prediction)
            for i in range(len(self.layers)-1): #last layer no error
                errors.append(self.layers[i].error_calculation(predictions[i]))
            
            for i in range(len(self.layers)):
                if i > 0 and i < len(self.layers)-2:  #First layer is fixed
                    self.layers[i].recalculate_state(0.01,errors[i],errors[i+1],self.W)
                elif i == len(self.layers)-2:
                    self.layers[i].recalculate_state(0.01,errors[i],errors[i],self.W)
                
            
            for i in range(len(self.layers)):
                if i == len(self.layers)-2:
                    self.W = matrix_update(0.01, errors[i], self.layers[i+1].state)
                    print("W",self.W)

        out = self.layers[len(self.layers)-1].state  #output of the last layer
        return out
            
    

# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) 

# Define the PCN model
input_size = 28*28  # MNIST image size
hidden_size = 10
num_layers = 10
#pcn_model = PCN([700,225,100,64])
pcn_model = PCN([64,64,64,64],W)


for epoch in range(5):
    for i, data in enumerate(trainloader):
        # data is a tuple of (inputs, labels)
        inputs, labels = data
        pcn_model.forward(inputs)
        

        # If you only want the first batch, you can break after the first iteration
       

# Train the PCN model
'''for epoch in range(5):  # Train for 5 epochs (you can increase it for better performance)
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        print(inputs.shape)
        print(labels.shape)
        inputs = inputs.flatten(1)  # Flatten the input images, look this up again
        
        #labels = labels.float()
        #print(labels.shape)
        
        #optimizer.zero_grad()
        
        # Forward pass
        outputs = pcn_model(inputs)
        print(outputs.shape)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        #optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        losses.append(running_loss / len(trainloader))'''

# Visualize the training loss

print('Finished Training')