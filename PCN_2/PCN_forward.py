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
import torch.nn.functional as F
from pathlib import Path


run = neptune.init_run(
    project="razergator/PC-Bachelor",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjI3YmUzMS04OTczLTQ5OWQtYjA0My1jOTEzMDljYTE0ZDkifQ==",
    capture_hardware_metrics=False, monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)  # JR: minimal logging for now
#
# run = neptune.init_run(
#     project="tns/scratch", capture_hardware_metrics=False, monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)  # JR: logging to JR neptune


DATA_DIR = Path(__file__).parents[1] / 'data'
DATA_DIR.mkdir(exist_ok=True)
model_path = DATA_DIR / 'pcn_model.pth'  # JR: for consistent paths

'''def visualize_predictions(loader, model, num_images=1): #Prediction speichern/abrufen, 
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = torch.flatten(inputs, start_dim=1)
    outputs, imagepredictions = model(images, batch_size=64) #outputs, imagepredictions = ; 784 -> 28x28 -> 1 x 28 x 28
    img = imagepredictions[0].view(1,28,28)
    img = img.permute(1,2,0)
    # Plotting
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    #for idx in range(num_images):
    ax = axes
    #img = images[0].permute(1, 2, 0)  # Reorder dimensions for plotting
    ax.imshow(img.numpy())
    ax.set_title(f'Predicted: {predicted[0]}') # 0 = idx
    ax.axis('off')
    plt.show() #vlt einrücken'''
def visualize_predictions(inputs, model, labels=None, num_images=5):
    model.eval()
    
    with torch.no_grad():
        outputs, imagepredictions = model(inputs, batch_size=64, fixed=True, logits=None)
    
    images = inputs.view(-1, 1, 28, 28).cpu()
    
    
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).squeeze()
        ax.imshow(img, cmap='gray')
        if labels is not None:
            ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()



def activation_function_prime(x):
    #return 1 - torch.tanh(x)**2 #tanh derivative, used with xavier initialization
    #return (x > 0).float()  # relu derivative
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1 - sigmoid)  # sigmoid derivative


class PCNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PCNLayer, self).__init__()
        self.output = output_size
        self.activationfunction = nn.Sigmoid() #nn.ReLU(), nn.Sigmoid()

    def predict(self, W):
        relu_state = self.activationfunction(self.state)
        prediction = relu_state @ W.T
        return prediction

    def error_calculation(self, prediction):
        error = self.state - prediction
        return error

    def recalculate_state(self, learning_rate, e_0, e_1, W,
                          last): 
        phi_prime = activation_function_prime(self.state)

        if last == True:
            pass
            e_1_reshaped = e_1.unsqueeze(2)
            W_t = W.T.unsqueeze(0)
            result_matrix_mult = torch.matmul(W_t, e_1_reshaped).squeeze(2)

            phi_prime_reshaped = phi_prime.unsqueeze(2)
            result_hadamard = phi_prime_reshaped * result_matrix_mult.unsqueeze(2)
            self.state += learning_rate * (result_hadamard.squeeze(2))
            
        else:
            e_1_reshaped = e_1.unsqueeze(2)  # Shape: (bs, 784, 1)
            W_t = W.T.unsqueeze(0)  # Shape: (1, 784, 100)
            result_matrix_mult = torch.matmul(W_t, e_1_reshaped).squeeze(2)  # Shape: (bs, 100)
            # Perform the Hadamard product with phi_prime
            phi_prime_reshaped = phi_prime.unsqueeze(2)  # Shape: (bs, 100, 1)
            result_hadamard = phi_prime_reshaped * result_matrix_mult.unsqueeze(2)

            self.state += (learning_rate * (-e_0 + result_hadamard.squeeze(2)))  # torch.matmul(e_0, W)

        



# Define the PCN class
class PCN(nn.Module): #Netzwerk "rückwärts" aufbauen, von letztem Layer zum ersten Layer, Error per Layer anschauen
    def __init__(self, hidden_sizes, W=None):
        super(PCN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.errors = []
        self.layer_states = []
        self.activationfunction = nn.Sigmoid() #nn.ReLU(), nn.Sigmoid()

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
                #torch.nn.init.kaiming_normal_(self.W[i], mode='fan_in', nonlinearity='relu') #He Init. 
               
        else:
            self.W = W
        

    def forward(self, input_data, batch_size, fixed=False, logits=None, update_rate=0.001, val = False):
        cycles = 60
        self.layer_states = []
        
        for i in range(len(self.layers)):
            state_tensor = torch.Tensor(batch_size, self.hidden_sizes[i])
            torch.nn.init.xavier_uniform_(state_tensor)
            self.layers[i].state = state_tensor #xavier initilization
            #n_in = self.hidden_sizes[i - 1]
            #he_std = math.sqrt(2. / n_in)
            #self.layers[i].state = torch.rand(batch_size, self.hidden_sizes[i]) * he_std
         
        self.layers[0].state = input_data
        if logits is not None:
            self.layers[-1].state = logits

        for cycle in range(cycles):
            predictions = [] #pred[0]
            self.errors = []
            loss_values = []
            # First all the predictions are calculated, then the errors and then the states are recalculated, so I don't have to access multiple layers at once
            for i, layer in enumerate(self.layers[1:], start=1):  # first layer no prediciton
                prediction = layer.predict(self.W[i])
                predictions.append(prediction)

            for i in range(len(self.layers) - 1):  
                self.errors.append(self.layers[i].error_calculation(predictions[i]))

            for i in range(1, len(self.layers)):
                if i == len(self.layers) - 1:
                    if fixed is False:
                        self.layers[i].recalculate_state(update_rate, e_0=0, e_1=self.errors[i - 1], W=self.W[i],
                                                     last=True)
                else:
                    self.layers[i].recalculate_state(update_rate, e_0=self.errors[i], e_1=self.errors[i - 1],
                                                     W=self.W[i], last=False)

            for i in range(len(self.layers)):
                self.layer_states.append(self.layers[i].state)

            for error in self.errors:
                error_flat = error.view(-1)
                se = (1 / 2) * torch.sum(error_flat ** 2)
                loss_values.append(se/error_flat.shape[0])  # 

           
            final_loss = torch.mean(torch.tensor(loss_values))
            #run["inference/loss"].append(final_loss)
        if val:
            run["val/loss"].append(final_loss)
        else:
            run["train/loss"].append(final_loss)
        out = self.layers[len(self.layers) - 1].state  # output of the last layer
        
        
        return out, predictions[0]
       

    def matrix_recalc(self, layer, learning_rate=0.01):  #
        if (layer < len(self.layers) - 1):  
            relulayer = self.activationfunction(self.layers[layer + 1].state)
            self.W[layer + 1] += learning_rate * (self.errors[layer].T @ relulayer)/self.errors[0].shape[0] 
            

            
    
    def inference(self, input):
        predictions = []
        self.layers[0].state = input
        for i, layer in enumerate(self.layers[1:], start=1):  # first layer no prediciton
                prediction = layer.predict(self.W[i])
                predictions.append(prediction)
        return predictions[0]


        

# Define the MNIST image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
batch_size = 64
trainset = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform) # JR: fixed path for dataset
valset = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=True)
# validation loader, generel testing, modelle trainieren, toplayer_fixed

# Define the PCN model
input_size = 28 * 28  # MNIST image size
pcn_model = PCN([28 * 28, 100, 100, 10])
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
correct_predictions = 0
total_predictions = 0
update_rate = 0.001


#with torch.no_grad():
for epoch in range(1): 
    pcn_model.train()
    for i, data in enumerate(trainloader):
        # data is a tuple of (inputs, labels)
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1)
        outputs,_ = pcn_model(inputs, batch_size, fixed=True, logits=F.one_hot(labels, num_classes=10).float())
        for j in range(len(pcn_model.layers)):
            pcn_model.matrix_recalc(j)
        print("Epoch: ", epoch, "Batch: ", i)
    #if epoch == 49:
    #    for i in range(2):
    #        visualize_predictions(inputs, pcn_model, labels)


    val_loss = 0.0
    mse_loss = 0.0
    pcn_model.eval()
    for data in valloader:
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1)
        outputs,predictions = pcn_model(inputs, batch_size, fixed=False, val = True)
        
        loss = criterion(outputs, labels)
        loss_2 = mse(predictions, inputs) 
        val_loss += loss.item()
        mse_loss += loss_2.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        run["val/CEL/loss"].append(loss.item())
        run["val/MSE/loss"].append(loss_2.item())

    avg_val_loss = val_loss / len(valloader)
    avg_mse_loss = mse_loss / len(valloader)
    run["val/CEL/avg_loss"].append(avg_val_loss)
    run["val/MSE/avg_loss"].append(avg_mse_loss)
    accuracy = 100 * correct_predictions / total_predictions
    run["val/accuracy"].append(accuracy)


print('Finished Training')
torch.save(pcn_model.state_dict(), model_path)
run['model_weights'].upload(str(model_path))
#print(f'Model saved to {model_path}')
run.stop()
