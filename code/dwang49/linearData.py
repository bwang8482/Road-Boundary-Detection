# CS 446 Machine Learning Final Project
# Road Boundary Detection via Machine Learning

# Apply Linear Learning
import matplotlib.pyplot as plt
import numpy as np

# Function To Load Data
def loadData():
    # Load Data
    data = np.load('data.npy')
    
    # Display Message
    print('Data Successfully Loaded!')
    
    return data

# Functions For Linear Learning
# Test Function
def test(data,label,w,theta):
    
    # Initialize Accuracy
    accuracy = 0;
    # Loop Through all data
    for row in range(data.shape[0]):
        if (np.dot(w,np.transpose(data[row,:])) + theta) >= 0:
            predict = 1
        else:
            predict = -1
        if predict == label[row]:
            accuracy += 1
    # Return
    return accuracy / data.shape[0]

# Simple Perceptron Algorithm
def perceptron(data,label,times = 20,debug = 0):
    # Initialization
    w = np.zeros(data.shape[1]);
    theta = 0;
    if debug == 1:
        cdf = np.zeros(data.shape[0])
    elif debug == 2:
        lasterror = -1
    # Loop through data several times
    for i in range(times):
        for row in range(data.shape[0]):
            # Check for Mistakes
            if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) <= 0:
                # Update with Learning Rate = 1
                w = w + label[row] * data[row,:]
                theta = theta + label[row]
                if debug == 1:
                    cdf[row] = 1
                elif debug == 2:
                    lasterror = i*data.shape[0] + row
            elif debug == 2:
                if i*data.shape[0] + row - lasterror >= 1000:
                    return i*data.shape[0] + row
                    
    if debug == 1:
        return np.cumsum(cdf)
    elif debug == 2:
        return -1
    else:
        # Return Results
        return w,theta;

# Perceptron with Margin
def perceptron_margin(data,label,eta,times = 20,debug=0):
    # Initialization
    w = np.zeros(data.shape[1]);
    theta = 0;
    if debug == 1:
        cdf = np.zeros(data.shape[0])
    elif debug == 2:
        lasterror = -1
    # Loop through data several times
    for i in range(times):
        for row in range(data.shape[0]):
            # Check for Mistakes with gamma = 1
            if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) < 1:
                if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) <= 0:
                    if debug == 1:
                        cdf[row] = 1
                    elif debug == 2:
                        lasterror = i*data.shape[0] + row
                # Update with Learning Rate eta
                w = w + eta * label[row] * data[row,:]
                theta = theta + eta * label[row]
            elif debug == 2:
                if i*data.shape[0] + row - lasterror >= 1000:
                    return i*data.shape[0] + row                
    if debug == 1:
        return np.cumsum(cdf)
    elif debug == 2:
        return -1
    else:
        # Return Results
        return w,theta;

# Simple Winnow Algorithm
def winnow(data,label,alpha,times = 20,debug=0):
    # Initialization
    w = np.ones(data.shape[1]);
    theta = -data.shape[1];
    if debug == 1:
        cdf = np.zeros(data.shape[0])
    elif debug == 2:
        lasterror = -1
    # Loop through data several times
    for i in range(times):
        for row in range(data.shape[0]):
            # Check for Mistakes
            if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) <= 0:
                # Update with parameter alpha
                w = w * np.power(np.ones(data.shape[1])*alpha,label[row]*data[row,:])
                if debug == 1:
                    cdf[row] = 1
                elif debug == 2:
                    lasterror = i*data.shape[0] + row
            elif debug == 2:
                if i*data.shape[0] + row - lasterror >= 1000:
                    return i*data.shape[0] + row
    if debug == 1:
        return np.cumsum(cdf)
    elif debug == 2:
        return -1
    else:
        # Return Results
        return w,theta

# Winnow with Margin
def winnow_margin(data,label,alpha,gamma,times = 20,debug=0):
    # Initialization
    w = np.ones(data.shape[1]);
    theta = -data.shape[1];
    if debug == 1:
        cdf = np.zeros(data.shape[0])
    elif debug == 2:
        lasterror = -1
    # Loop through data several times
    for i in range(times):
        for row in range(data.shape[0]):
            # Check for Mistakes with margin = gamma
            if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) < gamma:
                if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) <= 0:
                    if debug == 1:
                        cdf[row] = 1
                    elif debug == 2:
                        lasterror = i*data.shape[0] + row
                # Update with parameter alpha
                w = w * np.power(np.ones(data.shape[1])*alpha,label[row]*data[row,:])
            elif debug == 2:
                if i*data.shape[0] + row - lasterror >= 1000:
                    return i*data.shape[0] + row

    if debug == 1:
        return np.cumsum(cdf)
    elif debug == 2:
        return -1
    else:
        # Return Results
        return w,theta    

# AdaGrad Algorithm
def adagrad(data,label,eta,times = 20,debug=0):
    # Initialization
    w = np.zeros(data.shape[1]);
    theta = 0;
    G_w = np.zeros(data.shape[1]);
    G_theta = 0;
    if debug == 1:
        cdf = np.zeros(data.shape[0])
    elif debug == 2:
        lasterror = -1
    # Loop through data several times
    for i in range(times):
        for row in range(data.shape[0]):
            # Check for Mistakes with margin = 1
            if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) <= 1:
                if label[row] * (np.dot(w,np.transpose(data[row,:])) + theta) <= 0:
                    if debug == 1: 
                        cdf[row] = 1
                    elif debug == 2:
                        lasterror = i*data.shape[0] + row
                G_w += data[row,:] * data[row,:]
                temp = G_w
                temp[temp==0] += 1
                w += eta * label[row] * data[row] / np.sqrt(temp)
                # Update theta
                G_theta += 1
                theta += eta * label[row] / np.sqrt(G_theta)
            elif debug == 2:
                if i*data.shape[0] + row - lasterror >= 1000:
                    return i*data.shape[0] + row
    if debug == 1:
        return np.cumsum(cdf)
    elif debug == 2:
        return -1
    else:
        # Return Results
        return w,theta



# Main Function Starts here
data = loadData()

rawdata = np.copy(data)
# Shuffle
np.random.shuffle(rawdata)
# Currently Just Stupid Split
dataNum = rawdata.shape[0]//5

trainData = rawdata[:dataNum*4,2:64]
trainLabel = rawdata[:dataNum*4,64]

testData = rawdata[dataNum*4:,2:64]
testLabel = rawdata[dataNum*4:,64]

"""
#%% Perceptron Simple
w,theta = perceptron(trainData,trainLabel,times=100)
accuracy = test(testData,testLabel,w,theta)

#%% Perceptron with Margin
maxaccuracy = 0
for eta in [1.5,0.25,0.03,0.005,0.001,0.0001]:
    w,theta = perceptron_margin(trainData,trainLabel,eta)
    accuracy = test(testData,testLabel,w,theta)
    if accuracy > maxaccuracy:
        maxaccuracy = accuracy
        maxeta = eta

#%% Winnow Simple
maxaccuracy = 0
for alpha in [1.1,1.01,1.005,1.0005,1.0001]:
    w,theta = winnow(trainData,trainLabel,alpha)
    accuracy = test(testData,testLabel,w,theta)
    if accuracy > maxaccuracy:
        maxaccuracy = accuracy
        maxalpha = alpha

#%% Winnow with Margin
maxaccuracy = 0
for alpha in [1.1,1.01,1.005,1.0005,1.0001]:
    for gamma in [2.0,0.3,0.04,0.006,0.001]:
        w,theta = winnow_margin(trainData,trainLabel,alpha,gamma)
        accuracy = test(testData,testLabel,w,theta)
        if accuracy > maxaccuracy:
            maxaccuracy = accuracy
            maxalpha = alpha
            maxgamma = gamma
"""
#%% AdaGrad
maxaccuracy = 0
for eta in [1.5,0.25,0.03,0.005,0.001]:
    w,theta = adagrad(trainData,trainLabel,eta)
    accuracy = test(testData,testLabel,w,theta)
    if accuracy > maxaccuracy:
        maxaccuracy = accuracy
        maxeta = eta