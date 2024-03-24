import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import config
import dataset
import model

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Run on {device}\n")

def saveModel(model, path):
  # Save the model
    torch.save(model.state_dict(), path)
    
def trainer(model, criterion, optimizer, data_loader, epochs=10, path="./model.pth"):
    print("Start training")

    for epoch in range(epochs): # loop over the dataset multiple times
        model.train()
        model.to(device)

        for train_seq, test_seq in data_loader:
            # Move data to `device`
            train_seq, test_seq = train_seq.to(device), test_seq.to(device)  

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(train_seq)
            loss = criterion(output, test_seq)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print('Finished Training\n')

    # Save model
    saveModel(model, path)

def test(model, criterion, data_loader):
    print('Start testing')
    model.eval()
    model.to(device)
    total_loss = 0

    with torch.no_grad():
        for train_seq, test_seq in data_loader:
            train_seq, test_seq = train_seq.to(device), test_seq.to(device)  # Data to GPU
            output = model(train_seq)
            loss = criterion(output, test_seq)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(data_loader)}\n')


if __name__ == '__main__':
    dataset = dataset.ClosePriceDataset(config.dataset_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    rnn_model = model.RNNModel()

    # Optimizer
    optimizer = optim.Adam(rnn_model.parameters(), lr=config.lr)
    # MSE Loss Function
    criterion = nn.MSELoss()

    # Execute training and validating
    trainer(rnn_model, criterion, optimizer, data_loader, config.epochs, config.saved_model_path)
    test(rnn_model, criterion, data_loader)
