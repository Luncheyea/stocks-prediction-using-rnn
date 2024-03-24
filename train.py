import model
import dataset
import config

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Evaluate the rnn model for stocks prediction.')
parser.add_argument('-d', '--dataset', type=str, default=config.dataset_path, help='The csv file path of dataset')
parser.add_argument('-m', '--model', type=str, default=config.saved_model_path, help='The path of saved model')
parser.add_argument('-l', '--lr', type=float, default=config.lr, help='Learning rate')
parser.add_argument('-e', '--epochs', type=int, default=config.epochs, help='Epochs')
args = parser.parse_args()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print(f"Run on {device}\n")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"lr: {args.lr}")
    print(f"epochs: {args.epochs}")
    print("==========")
    
    close_price_dataset = dataset.ClosePriceDataset(args.dataset)
    data_loader = DataLoader(close_price_dataset, batch_size=1, shuffle=True)
    rnn_model = model.RNNModel()

    # Optimizer
    optimizer = optim.Adam(rnn_model.parameters(), lr=args.lr)
    # MSE Loss Function
    criterion = nn.MSELoss()

    # Execute training and testing
    trainer(rnn_model, criterion, optimizer, data_loader, args.epochs, args.model)
    test(rnn_model, criterion, data_loader)
