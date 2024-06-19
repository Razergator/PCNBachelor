import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def relu_prime(x): #ReLu derivative is 1 if x > 0, 0 otherwise
    return (x > 0).float()

class PCNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PCNLayer, self).__init__()
        self.output = output_size
        self.relu = nn.ReLU()

    def predict(self, W):
        relu_state = self.relu(self.state)
        prediction = relu_state @ W.T
        return prediction

    def error_calculation(self, prediction):
        #print("prediction:", prediction.shape)
        #print("State:", self.state.shape)
        #prediction = prediction.view(self.state.shape)
        error = self.state - prediction 
        
        return error

    def recalculate_state(self, learning_rate, e_0,e_1, W, i): #e_i = error, W = weight_matrix, phi_prime = derivative of an activation function(here ReLu)
        phi_prime = relu_prime(self.state)
        #print("phi_prime",phi_prime.shape)
        
        #
        
        if i == 0:
            pass#self.state = learning_rate * ((phi_prime))  @  (W.T @ e_1) 
        else:
            print("e_1",e_1.shape)
            print("e_0",e_0.shape)
            print("W.T",W.T.shape)
            print("W.T@e_1",(W.T @ e_1).shape)
            print("-e_0 + phi_prime",(-e_0 + (phi_prime)).shape)

            self.state = learning_rate * (-e_0 + (phi_prime))  * (W.T @ e_1) #torch.matmul(e_0, W)
        #print(self.state) 
       


# Define the PCN class
class PCN(nn.Module):
    def __init__(self, hidden_sizes, W = None):
        super(PCN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList() 
        self.errors = []
       
        for i in range(len(hidden_sizes)):
            if i == 0:
                #print("i=0",hidden_sizes[i])
                self.layers.append(PCNLayer((hidden_sizes[i]),1))
                #print("layer:",self.layers[i])
            if i > 0:
                #print("i>0",hidden_sizes[i],hidden_sizes[i-1])
                self.layers.append(PCNLayer(hidden_sizes[i],hidden_sizes[i-1]))
                #print("layer:",self.layers[i])
        #print("W_2")
        if W == None:
            self.W = [torch.ones((hidden_sizes[i-1],hidden_sizes[i])) for i in range(len(self.layers))]
            self.W[0] = None
            for i in range(1, len(self.layers)):
                torch.nn.init.xavier_uniform_(self.W[i])
                #print("Weight_matrix", self.W[i].shape)
        else:
            self.W = W
        
        

    def forward(self, input_data, batch_size):
        cycles = 10     
        
        for i in range(len(self.layers)):
            if i > 0:
                self.layers[i].state = torch.rand(batch_size,self.hidden_sizes[i])
                #print("sizes", self.layers[i].state.shape) 

        self.layers[0].state = input_data
        #print("state_0", self.layers[0].state.shape)
        

        for _ in range(cycles): 
            predictions = []
            self.errors = []
            #First all the predictions are calculated, then the errors and then the states are recalculated, so I don't have to access multiple layers at once
            for i, layer in enumerate(self.layers[1:], start=1): #first layer no prediciton
                prediction = layer.predict(self.W[i])
                predictions.append(prediction)
                
            #for i in range(len(predictions)):
            #   print("prediction",predictions[i].shape)

            for i in range(len(self.layers)-1): #last layer no error
                self.errors.append(self.layers[i].error_calculation(predictions[i]))
                #print("error",self.errors[i].shape)
            
            
            for i in range(1,len(self.layers)):
                if i < len(self.layers)-1:  #First layer is fixed
                    self.layers[i].recalculate_state(0.01,self.errors[i-1],self.errors[i],self.W[i],i)
               

        out = self.layers[len(self.layers)-1].state  #output of the last layer
        return out
                
    def matrix_recalc(self, layer, learning_rate=0.01):
        #print("error shape:", error.shape)
        #print("relu_prime(state_1).T shape:", relu_prime(state_1).T.shape)
        if(layer > 0 and layer < len(self.layers)-2):
            #print("self.errors[layer]",self.errors[layer].shape)   
            #print("relu_prime((self.layers[layer+1]).state).T",relu_prime((self.layers[layer+1]).state).T.shape)
            self.W[layer] = learning_rate * (self.errors[layer] * relu_prime((self.layers[layer+1]).state).T)
            #print("W_1",self.W[layer])
        
        if(layer == len(self.layers)-1):
            #self.W[layer] = learning_rate * (self.errors[layer-1] * relu_prime((self.layers[layer]).state).T)
            pass
            #print("W_2",self.W[layer])
        
        

    
            
    

# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True) 

# Define the PCN model
input_size = 28*28  # MNIST image size
pcn_model = PCN([28*28,100,100,10]) #Zwischenlayer müssen! mit der batchsize übereinstimmen, ich weiß aber nicht wieso


for epoch in range(1):
    for i, data in enumerate(trainloader):
        # data is a tuple of (inputs, labels)
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1)
        pcn_model.forward(inputs, batch_size)
        #pcn_model.matrix_recalc(2)
        #for j in range(len(pcn_model.layers)):
            #print("layer",j)
            #pcn_model.matrix_recalc(j)
        #print("inputs:",inputs.shape)
        #print("train",enumerate(trainloader))
        #print("i",i)

       

        # If you only want the first batch, you can break after the first iteration
        if i == 0: #last batch is (32,1,28,28) for some godforsaken reason and I can't get it to work, I think I might actually know why, maybe the amount of data is not divisible by 64, and the last batch is just the rest
            break
       


print('Finished Training')