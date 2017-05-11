# CS 446 Machine Learning Final Project
# Road Boundary Detection via Machine Learning

# Data Loading and Feature Extraction
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import perimeter
from skimage import io
import numpy as np

def loadData():
    # Load Data
    data = np.load('data.npy')
    
    # Display Message
    print('Data Successfully Loaded!')
    
    return data

def checkSlic():
    
    data = loadData()
    # Error List
    error = []
    
    for idx in range(159,160):
        
        # Display
        print('Processing Image #',idx+1,sep='')
        # Load Image and Ground Truth Image as Float
        img = img_as_float(io.imread('data/images/'+str(idx+1)+'.png'))
        img_gt = img_as_float(io.imread('data/gt_images/'+str(idx+1)+'.png'))
    
        # Perform SLIC
        img_seg = slic(img, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    
        img_sample = np.zeros(img_gt.shape[:-1])
        
        img_data = data[np.where(data[:,65]==(idx+1))]
        
        big = np.amax(img_seg)
        roi = (img_seg == big)
        roi_idx = np.nonzero(roi)
        biglabel = img_gt[int(np.mean(roi_idx[0])),int(np.mean(roi_idx[1])),2]
        
        for row in range(img_gt.shape[0]):
            for col in range(img_gt.shape[1]):
                label = img_seg[row,col]
                if label == big:
                    img_sample[row,col] == biglabel
                elif img_data[label][64] == 1:
                    img_sample[row,col] = 1
        
        curerror = sum(sum(img_sample!=img_gt[:,:,2]))/img_gt.shape[0]/img_gt.shape[1]
        error.append(curerror)
        
        plt.figure(1)
        fig1 = plt.imshow(mark_boundaries(img, img_seg))
        fig1.axes.get_xaxis().set_visible(False)
        fig1.axes.get_yaxis().set_visible(False)
        plt.figure(2)
        fig2 = plt.imshow(mark_boundaries(img_gt, img_seg))
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        plt.figure(3)
        fig2 = plt.imshow(img)
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        plt.figure(4)
        fig2 = plt.imshow(img_gt)
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        plt.figure(5)
        fig2 = plt.imshow(img_sample)
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)        
    return error

def checkLabel(idx):
    
    # Load Trained File
    trained = np.load('label'+str(idx)+'.npy')
    
    # Load Original file
    test = np.load('test'+str(idx)+'.npy')

    # Compute Accuracy
    accuracy =  sum(test[:,64] == trained)/test.shape[0]
    
    # Compute Road Not Recognized Error
    roadFail = sum(np.logical_and(test[:,64]==1, trained==0))/test.shape[0]
    roadNone = sum(np.logical_and(test[:,64]==0, trained==1))/test.shape[0]

    return accuracy, roadFail, roadNone

# Main Function Starts here

op = 2;

# Save Data to File
if op == 0:
    pass
# Load Data to Workspace
elif op == 1:
    data = loadData();
elif op == 2:
    error = checkSlic();
elif op == 3:
    accuracy, roadFail, roadNone = checkLabel(1);

