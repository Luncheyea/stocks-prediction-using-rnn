import re
import os
import pandas as pd
import matplotlib.pyplot as plt

def plotData(file_path):
    """
    Reads a file containing a title line followed by lines of data,
    each with a 'Date' and 'Close Price', and plots the 'Close Price' over 'Date'.
    
    Parameters:
    - file_path: The path to the file to read and plot.
    """
    # get the file name
    file_name = re.search(r'[^\/\\]+?$', file_path).group()

    # Read the file into a DataFrame, skipping the first row (title) and assuming comma separation
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['Date', 'Close Price'])
    
    # Convert 'Date' to datetime format for better plotting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close Price'], marker='o', linestyle='-', color='blue')
    plt.title(f"Close Price Over Time from '{file_name}'")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated date labels
    plt.show()

# Specify the path to the file
file_path = './dataset/split'
# file_path = './dataset/test'

# List all .csv files in the directory
files = os.listdir(file_path)
csv_files = [file for file in files if file.endswith('.csv')]

# Call the function to read the file and plot the data
plotData(f"{file_path}/{1}.csv")
# for i in range(1, len(csv_files) + 1):
#     plotData(f"{file_path}/{i}.csv")
