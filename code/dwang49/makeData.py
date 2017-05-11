# CS 446 Machine Learning Final Project
# Road Boundary Detection via Machine Learning

# Data Loading and Feature Extraction
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import perimeter
from skimage import io
import numpy as np
import random

# Read Images, Perform Slic, Save the Data
def saveData():
    # Number of Total Images
    imgNum = 289;
    # Bins used for Histogram
    bins = np.linspace(0,1,num=21)
    
    # Data Container: [centerx, centery, area, perimeter, rhist:20bin, ghist, bhist; label]
    data = [];
    
    
    # Start Extracting Data
    print('Start Reading Images...')
    
    # Start Main Loop
    for idx in range(imgNum):
        
        # Display Message
        print('Loading Image #',idx,sep='')    
    
        # Load Image and Ground Truth Image as Float
        img = img_as_float(io.imread('data/images/'+str(idx+1)+'.png'))
        img_gt = img_as_float(io.imread('data/gt_images/'+str(idx+1)+'.png'))
        
        # Perform SLIC
        img_seg = slic(img, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
        
        # Loop through all Segments
        for label in range(np.amax(img_seg)+1):
            
            # Display Message
            if label%100 == 0:
                print('Processing Image #:',idx+1,' Seg #:',label,sep='')
            
            # Create Entry for Segment
            data.append([])
            
            # Boolean Mask of Region of Interests
            roi = (img_seg == label)
            
            # Extract Index Information
            roi_idx = np.nonzero(roi)
            # Attach center x(row), center y(col)
            data[-1].append(np.mean(roi_idx[0]))
            data[-1].append(np.mean(roi_idx[1]))
    
            # Append Area
            data[-1].append(np.shape(roi_idx)[1])
            # Append Perimeter
            data[-1].append(perimeter(roi))
            # Extract Color Information
            roi_rgb = img[roi]
            # Adding R,G,B Channels
            for channel in range(3):
                hist,_ = np.histogram(roi_rgb[:,channel],bins,density=True)
                data[-1].extend(hist.tolist())
    
            # Adding Label
            if img_gt[int(data[-1][0]),int(data[-1][1]),2] > 0:
                data[-1].append(1)
            else:
                data[-1].append(0)
    
            # Append Data Flag
            data[-1].append(idx+1)
    # Output Data to file
    np.save('data.npy', data)
    
    # Display Message
    print('Data Successfully Saved!')

    return

def loadData():
    # Load Data
    data = np.load('data.npy')
    
    # Display Message
    print('Data Successfully Loaded!')
    
    return data

# Shuffle Indx
def randIdx():
        
    # Data Array
    idxs = list(range(1,290))
    
    # Random Seed
    random.seed()
    
    # Shuffle
    random.shuffle(idxs)
    
    # Cut to Pieces
    trainIdx = []
    testIdx = []
    length = len(idxs)//5
    for cvIdx in range(5):
        testIdx.append(idxs[cvIdx*length:cvIdx*length+length])
        trainIdx.append(idxs[0:cvIdx*length] + idxs[cvIdx*length+length:])
    
    np.save('trainIdx.npy', trainIdx)
    np.save('testIdx.npy', testIdx)
    # Write to File
    print('Random Indexes  Saved!')

    return

# Random Data
def randData():
    
    # Display
    print('Start Processing 5 Fold Validation Data...')
    
    # Load Data
    data = loadData()
    
    # Load Saved Index
    trainIdx = np.load('trainIdx.npy')
    testIdx = np.load('testIdx.npy')
    
    # 5 Fold
    for idx in range(5):
        print('Processing Data Fold #',idx+1,sep='')
        # Create Mask
        trIdx = trainIdx[idx]
        teIdx = testIdx[idx]

        # Paste TrainData        
        trainData = data[np.where(data[:,65] == trIdx[0])]
        for i in trIdx[1:]:
            trainData = np.concatenate((trainData, data[np.where(data[:,65] == i)]), axis=0)
        
        # Paste TestData
        testData = data[np.where(data[:,65] == teIdx[0])]
        for i in teIdx[1:]:
            testData = np.concatenate((testData, data[np.where(data[:,65] == i)]), axis=0)        
    
        # Save to File
        np.save('train'+str(idx+1)+'.npy', trainData)
        np.save('test'+str(idx+1)+'.npy', testData)
        
        print('Successfully Saved Data Fold #',idx+1,' !',sep='')
    
    return

# Load Fold Data
def loadFold(idx):
    
    trainData = np.load('train'+str(idx)+'.npy')
    testData = np.load('test'+str(idx)+'.npy')
    
    return trainData, testData

# Main Function Starts here
# Operation Flag: 
#    0 for Generate Data;
#    1 for Load Data
#    2 for Save Random Idexes
#    3 for Save 5 Fold Data
#    4 for Validate Fold Data
op = 0;

# Save Data to File
if op == 0:
    saveData();
# Load Data to Workspace
elif op == 1:
    data = loadData();
# Save Random Idexes
elif op == 2:
    randIdx();
# Save Random Data
elif op == 3:
    randData();
# Load Fold Data
elif op == 4:
    trainData, testData = loadFold(2);