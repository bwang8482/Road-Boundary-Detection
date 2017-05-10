# CS 446 Machine Learning Final Project
# Road Boundary Detection via Machine Learning

# Data Loading and Feature Extraction
import numpy as np


names = ['train1', 'train2', 'train3', 'train4', 'train5', 'test1', 'test2', 'test3', 'test4', 'test5']

for name in names:
    # Load Data
    data = np.load(name + '.npy')

    # Display Message
    print(name + ' Successfully Loaded!')

    # Main Function Starts here
    np.savetxt(name + '.out', data, delimiter=',', fmt='%.4f')
    print('Data Successfully Saved!')
