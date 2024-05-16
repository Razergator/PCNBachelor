import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


losses = [] #list for visualizing loss

#defining all necessary methods here, maybe I need to put them into PCNLayer class, unsure yet
def predict(layer_i):
        prediction = torch.random() #placeholder
        return prediction

def error_calculation(layer_previous_state, prediction):
        error = layer_previous_state - prediction
        return error

def recalculate_state(layer_i_state,error):
        state = layer_i_state + error #placeholder
        return state


# Define the PCNLayer class, maybe I only need to have self.state as a tensor and not the other variables
class PCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PCNLayer, self).__init__()
        self.prediction = nn.Linear(hidden_size, input_size)
        self.error = nn.Linear(input_size, hidden_size)
        self.state = torch.random(hidden_size) #state is just the tensor that is saved in the layer
        self.hidden_size = hidden_size

    '''def forward(self, input_data, feedback):
        # Initialize feedback with zeros
        feedback = torch.zeros(self.hidden_size)
        # Forward pass (error computation)
        prediction_error = input_data - feedback
        # Update feedback using error
        feedback = self.error(prediction_error)
        
        return feedback, prediction_error'''
    
    '''def backward(self, feedback):
        # Initialize prediction with zeros
        prediction = torch.zeros(self.hidden_size)
        # Backward pass (prediction computation)   
        prediction = self.prediction(feedback)

        return prediction''' #most likely unnecessary, I dont think I need the forward methode, due to the cyclic nature of the PCN,
        #I can just call the predict method in the forward pass of the PCN class as needed, Ill still leave it in for now
    
    
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
        feedback = torch.zeros_like(input_data)  # Initialize feedback signal
        
        # Forward pass through PCN layers
        for _ in range(10): #first try at getting it cyclic, right now it just repeats multiple times
            for i, layer in enumerate(self.layers):
                #feedback = self.relu(feedback) #add ReLu layer for non-linearity

                if i > 0 and i < len(self.layers) - 1: #if not first or last layer
                    layer_i = self.layers[i]
                    layer_previous = self.layers[i-1]
                    prediction = predict(layer_i)
                    error = error_calculation(layer_previous.state,prediction)
                    layer_i.state = recalculate_state(layer_i.state, error) #update state of layer i
                    '''
                    prediction = layer_i.predict()
                    error = layer_previous.state(prediction)
                    layer_i.recalculate_state(error)
                    self.relu(layer_i.state) #add ReLu layer for non-linearity
                    '''

       
     
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

# Visualize the training loss
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print('Finished Training')