# The csv file path of training data
dataset_path = './dataset/0050.TW close.csv'

# The path for saved model
saved_model_path = './model.pth'

# The number of consecutive training data in each epoch
sequence_length = 30

# In each epoch, the first `number_of_training` data are used as training data, 
# while the remaining data are considered as ground truth
# This value should not be greater than or equal to `sequence_length`
number_of_training = 25
assert number_of_training < sequence_length, \
    '`config.number_of_training` should not be greater than or equal to `config.sequence_length`'

# Epoch
epochs = 10

# Learning rate
lr = 0.001
