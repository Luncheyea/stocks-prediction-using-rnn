import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, num_layers=1, output_size=5):  # `output_size` corresponds to test sequence length
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1  = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2  = nn.ReLU()

         # Output layer to match test sequence length
        self.fc3 = nn.Linear(hidden_size, output_size) 
    
    def predict(self, input):
        with torch.no_grad():
            return self.forward(input).item()

    def forward(self, x):
        batch_size = x.size(0)
        x, _ = self.rnn(x.unsqueeze(-1))
        
        x = x[:, -1, :]  # Take the last output of RNN to feed into the FC layer
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        
        return self.fc3(x)