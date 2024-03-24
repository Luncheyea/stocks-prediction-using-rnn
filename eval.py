import train
import model
import dataset
import config

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Evaluate the rnn model for stocks prediction.')
parser.add_argument('-d', '--dataset', type=str, default=config.dataset_path, help='The csv file path of dataset')
parser.add_argument('-m', '--model', type=str, default=config.saved_model_path, help='The path of model')
args = parser.parse_args()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def printTensorData(tensor, data_each_line):
    """
    Prints the data in the tensor, with each line containing up to `data_each_line` data points.
    
    Parameters:
    - tensor: The PyTorch tensor to print.
    - data_each_line: each line containing up to `data_each_line` data points.
    """

    # Ensure tensor is detached and moved to CPU
    data = tensor.detach().cpu().numpy()
    # Reshape the data to 1D for simplicity
    data = data.reshape(-1)
    # Iterate and print every `data_each_line` elements
    for i in range(0, len(data), data_each_line):
        print(data[i:i+data_each_line])

def evaluater(model, criterion, data_loader):
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print("==========")
    print('Start evaluating')

    model.eval()
    model.to(device)

    with torch.no_grad():
        for input_seq, ground_truth_seq in data_loader:
            # evaluate the data
            input_seq, ground_truth_seq = input_seq.to(device), ground_truth_seq.to(device)  # Data to GPU
            output = model(input_seq)
            loss = criterion(output, ground_truth_seq)
            
            # print the result
            each_line_data = 5
            print('Input data:')
            printTensorData(input_seq, each_line_data)
            print('Ground truth:')
            printTensorData(ground_truth_seq, each_line_data)
            print('Output data:')
            printTensorData(output, each_line_data)
            print(f'Loss: {loss.item()}')
            print('==============================\n')
    print('Finish evaluating')

if __name__ == '__main__':
    print(f"Run on {device}\n")

    # Load the dataset
    close_price_dataset = dataset.ClosePriceDataset(args.dataset)
    data_loader = DataLoader(close_price_dataset, batch_size=1, shuffle=True)
    
    # Load the model state dict
    rnn_model = model.RNNModel()
    rnn_model.load_state_dict(torch.load(args.model))

    # MSE Loss Function
    criterion = nn.MSELoss()

    # Execute evaluation
    evaluater(rnn_model, criterion, data_loader)
