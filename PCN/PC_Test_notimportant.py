import torch
import torch.nn as nn

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

# Example usage:
input_size = 10
hidden_size = 10
num_layers = 3

pcn_model = PCN(input_size, hidden_size, num_layers)

# Generate random input data
input_data = torch.randn(1, input_size)

# Forward pass through PCN
prediction = pcn_model(input_data)

print("Prediction:", prediction)
