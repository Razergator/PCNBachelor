import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import neptune
import math

run = neptune.init_run(
    project="razergator/PC-Bachelor",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjI3YmUzMS04OTczLTQ5OWQtYjA0My1jOTEzMDljYTE0ZDkifQ==",
)

model_path = './pcn_model.pth'

def relu_prime(x):  # ReLu derivative is 1 if x > 0, 0 otherwise
    #return 1 - torch.tanh(x)**2 #tanh derivative
    return (x > 0).float() #relu derivative


class PCNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PCNLayer, self).__init__()
        self.output = output_size
        self.relu = nn.ReLU()

    def predict(self, W):
        #print("self.state",self.state)
        relu_state = self.relu(self.state)
        prediction = relu_state @ W.T
        #print("prediction",prediction)
        return prediction

    def error_calculation(self, prediction):
        error = self.state - prediction
        #print(error)
        return error

    def recalculate_state(self, learning_rate, e_0, e_1, W, last):  # e_i = error, W = weight_matrix, phi_prime = derivative of an activation function(here ReLu)
        phi_prime = relu_prime(self.state)
       
        if last == True:
            pass
            #print("e_1", e_1)
            e_1_reshaped = e_1.unsqueeze(2)
            #print("e_1_reshaped", e_1_reshaped)
            #print("W_T_before", W.T)
            W_t = W.T.unsqueeze(0)
            #print("W_T_after", W_t)
            result_matrix_mult = torch.matmul(W_t, e_1_reshaped).squeeze(2)
            #print("result_matrix_mult",result_matrix_mult)

            phi_prime_reshaped = phi_prime.unsqueeze(2)
            result_hadamard = phi_prime_reshaped * result_matrix_mult.unsqueeze(2)

            self.state += learning_rate * (result_hadamard.squeeze(2))
            print("Works")
            #print(self.state)

        else:
            pass
            e_1_reshaped = e_1.unsqueeze(2)  # Shape: (bs, 784, 1)
            W_t = W.T.unsqueeze(0)  # Shape: (1, 784, 100)
            result_matrix_mult = torch.matmul(W_t, e_1_reshaped).squeeze(2)  # Shape: (bs, 100)

            # Perform the Hadamard product with phi_prime
            phi_prime_reshaped = phi_prime.unsqueeze(2)  # Shape: (bs, 100, 1)
            result_hadamard = phi_prime_reshaped * result_matrix_mult.unsqueeze(2)

            self.state += (learning_rate * (-e_0 + result_hadamard.squeeze(2))) # torch.matmul(e_0, W)
            
        # print(self.state)

    # e_0[0].unsqueeze(1)
    # phi_prime[0].unsqueeze(1) * W.T *


# Define the PCN class
class PCN(nn.Module):
    def __init__(self, hidden_sizes, W=None):
        super(PCN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.errors = []
        self.layer_states = []

        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(PCNLayer((hidden_sizes[i]), 1))
            if i > 0:
                self.layers.append(PCNLayer(hidden_sizes[i], hidden_sizes[i - 1]))
        if W == None:
            self.W = [torch.ones((hidden_sizes[i - 1], hidden_sizes[i])) for i in range(len(self.layers))]
            self.W[0] = None
            for i in range(1, len(self.layers)):
                torch.nn.init.xavier_uniform_(self.W[i])
        else:
            self.W = W

    def forward(self, input_data, batch_size, fixed = False): #add fixed parameter for toplayer, fixed im training, not fixed in inference, not fixed rest
        cycles = 10000
        self.layer_states = []
        

        for i in range(len(self.layers)):
            if i > 0:
                n_in = self.hidden_sizes[i - 1]
                he_std = math.sqrt(2. / n_in)
                self.layers[i].state = torch.rand(batch_size, self.hidden_sizes[i]) * he_std #He initilization
                #state_tensor = torch.Tensor(batch_size, self.hidden_sizes[i])
                #torch.nn.init.xavier_uniform_(state_tensor)
                #self.layers[i].state = state_tensor #xavier initilization
                



        self.layers[0].state = input_data

        for _ in range(cycles):
            predictions = []
            self.errors = []
            loss_values = []
            # First all the predictions are calculated, then the errors and then the states are recalculated, so I don't have to access multiple layers at once
            for i, layer in enumerate(self.layers[1:], start=1):  # first layer no prediciton
                #print("self.W[i]",self.W[i])
                prediction = layer.predict(self.W[i])
                #print("prediction",prediction)
                predictions.append(prediction)


            for i in range(len(self.layers) - 1):  # last layer no error
                self.errors.append(self.layers[i].error_calculation(predictions[i]))
                #print("self.errors",self.errors[i])
                #print("self.layers[i].state",self.layers[i])
                #print("predictions[i]",predictions[i])
                #print(self.errors[i])


            for i in range(1, len(self.layers)-1):
                #pass
                #print("i",i)
                if fixed == False and i == len(self.layers)-1:
                   self.layers[i].recalculate_state(0.1, e_0=0, e_1=self.errors[i-1], W=self.W[i], last = True)
                    #print("i",i)
                    #print("e_1",self.errors[i-1])
                else:
                    self.layers[i].recalculate_state(0.1, e_0=self.errors[i], e_1=self.errors[i-1], W=self.W[i], last = False)
                    #run["states"].append(self.layers[i])
                    #print("self.layers", self.layers[i].state)

            for i in range(len(self.layers)):
                self.layer_states.append(self.layers[i].state)
            
            for error in self.errors:
                # Flatten the error tensor
                error_flat = error.view(-1)
                # Calculate a metric, e.g., mean squared error
                mse = torch.mean(error_flat ** 2)
                # Append the scalar value to the list
                loss_values.append(mse)

            # if top layer is not fixed as during inference, we need to update top layer as well, add eq(5) for top layer (l=max), e_+1 has to be e_l-1
            # if fixed: ...
            final_loss = torch.mean(torch.tensor(loss_values))
            run["loss"].append(final_loss)
            #print("final_loss",final_loss)
        
        out = self.layers[len(self.layers) - 1].state  # output of the last layer
        #print("self.layers",self.layers[len(self.layers)-1].state)
        return out

    def matrix_recalc(self, layer, learning_rate=0.01):
        if (layer < len(self.layers) - 2):
            self.W[layer+1] = learning_rate * (self.errors[layer].T @ relu_prime((self.layers[layer + 1]).state)) 



# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True) #validation loader, generel testing, modelle trainieren, toplayer_fixed

# Define the PCN model
input_size = 28 * 28  # MNIST image size
pcn_model = PCN(
    [28 * 28, 100, 100, 10])  

with torch.no_grad():
    
    for epoch in range(1):
        for i, data in enumerate(trainloader):
            # data is a tuple of (inputs, labels)
            inputs, labels = data
            inputs = torch.flatten(inputs, start_dim=1)
            pcn_model.forward(inputs, batch_size, fixed = False)
            for j in range(len(pcn_model.layers)):
                #print("layer",j)
                pcn_model.matrix_recalc(j)
            #print("inputs:",inputs.shape)
            #print("train",enumerate(trainloader))
            #print("i",i)
            #states = pcn_model.layer_states
            #for state in states:
            #    test = state
                #print("state",state)
                #run["states"].append(state)
            #run["states"].append(states)

            # If you only want the first batch, you can break after the first iteration
            if i == 0:
              break
    
#run.stop()
print('Finished Training')
torch.save(pcn_model.state_dict(), model_path)
run['model_weights'].upload(model_path)
print(f'Model saved to {model_path}')
run.stop()

#training
#überprüfen der Inidzes, validation of network
#speichern und laden von netzwerken
#Donnerstag 11:30 meeting
#xavier + tanh looks pretty bad
#relu + he looks pretty good