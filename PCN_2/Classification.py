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
    def __init__(self, input_size, output_size, batch_size = 64):
        super(PCNLayer, self).__init__()
        self.output = output_size
        #self.state = torch.rand(batch_size, output_size)#state is just the tensor that is saved in the layer
        #self.state = torch.rand(batch_size, hidden_size)
        self.linear = nn.Linear(input_size, output_size, bias=False)#torch.ones((10, 10), requires_grad=True)
        self.relu = nn.ReLU()
        #self.hidden_size = hidden_size
    
    
    
    def predict(self):
        relu_state = self.relu(self.state)
        #print("State:", self.state.shape)
        #print("self.linear:", relu_state.shape)
        prediction = self.linear(relu_state)
        #print("Prediction:", prediction.shape)
        return prediction

    def error_calculation(self, prediction):
        print("prediction:", prediction.shape)
        print("State:", self.state.shape)
        error = self.state - prediction
        
        return error

    def recalculate_state(self,error):
        print("State",self.state.shape)
        print("Error", error.shape)
        self.state = self.state - error #brain inspired computation via predictive coding
        #print(self.state)

        #return self.state


    
    
# Define the PCN class
class PCN(nn.Module):
    def __init__(self, hidden_sizes):
        super(PCN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                print("i=0",hidden_sizes[i])
                self.layers.append(PCNLayer(hidden_sizes[i],1))
                print("layer:",self.layers[i])
            if i > 0:
                print("i>0",hidden_sizes[i],hidden_sizes[i-1])
                self.layers.append(PCNLayer(hidden_sizes[i],hidden_sizes[i-1]))
                print("layer:",self.layers[i])

        
        #self.classifier = nn.Linear(hidden_size, num_classes)
    
         
        
        
        
    def forward(self, input_data):
        cycles = 10  #Number is subject to change
        #feedback = self.relu(feedback) #add ReLu layer for non-linearity
        #self.layers[0].state = input_data, dk how to fix rn
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
                if i > 0 and i < 3:  #First layer is fixed
                    self.layers[i].recalculate_state(errors[i])
        
        out = self.layers[len(self.layers)-1].state  #output of the last layer 

        return out
            
       
       
        

   

# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) #is normally 64, i dont really get it to work now

# Define the PCN model
input_size = 28*28  # MNIST image size
hidden_size = 10
num_layers = 10
pcn_model = PCN([700,100,100,100])
for i, data in enumerate(trainloader):
    # data is a tuple of (inputs, labels)
    inputs, labels = data
    pcn_model.forward(inputs)
    # Now you can use inputs and labels
    print(inputs)
    print(labels)

    # If you only want the first batch, you can break after the first iteration
    if i == 0:
        break
 #just to test if it works, add input data instead of trainloader

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(pcn_model.parameters(), lr=0.01)

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