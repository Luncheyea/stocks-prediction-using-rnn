import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Data loader
class ClosePriceDataset(Dataset):
    def __init__(self, file_path, sequence_length=30, number_of_training=25):
        self.data = pd.read_csv(file_path)['Close Price'].values
        self.sequence_length = sequence_length
        self.number_of_training = number_of_training
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        random_start = np.random.randint(0, len(self.data) - self.sequence_length)
        sample = self.data[random_start:random_start + self.sequence_length]
        train_sample = sample[:int(self.number_of_training)]
        test_sample = sample[int(self.number_of_training):]
        return torch.tensor(train_sample, dtype=torch.float), torch.tensor(test_sample, dtype=torch.float)
    

# 2. Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, num_layers=1, output_size=5):  # `output_size` corresponds to test sequence length
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output layer to match test sequence length
    
    def forward(self, x):
        batch_size = x.size(0)
        x, _ = self.rnn(x.unsqueeze(-1))
        x = x[:, -1, :]  # Take the last output of RNN to feed into the FC layer
        x = self.fc(x)
        return x


# 3. Training part
def train(model, data_loader, optimizer, criterion, epochs=10):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        for train_seq, test_seq in data_loader:
            train_seq, test_seq = train_seq.to(device), test_seq.to(device)  # Move data to GPU
            optimizer.zero_grad()
            # Reshape train_seq for the RNN: [batch_size, sequence_length, n_features]
            output = model(train_seq)
            # The output is now aligned with test_seq size, so we can calculate loss
            loss = criterion(output, test_seq)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# 4. Testing part
def test(model, data_loader, criterion):
    model.eval()
    model.to(device)
    total_loss = 0

    with torch.no_grad():
        for train_seq, test_seq in data_loader:
            train_seq, test_seq = train_seq.to(device), test_seq.to(device)  # Data to GPU
            output = model(train_seq)
            loss = criterion(output, test_seq)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(data_loader)}')


dataset = ClosePriceDataset('./dataset/0050.TW close.csv')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
model = RNNModel()

# 5. Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 6. MSE Loss Function
criterion = nn.MSELoss()

# Execute training and testing
train(model, data_loader, optimizer, criterion)
test(model, data_loader, criterion)
