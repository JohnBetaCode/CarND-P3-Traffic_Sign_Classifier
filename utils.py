# =============================================================================
"""
Code Information:
    Date: 03/27/2019
	Programmer: John A. Betancourt G.
	Mail: john.betancourt93@gmail.com / john@kiwicampus.com
    Web: www.linkedin.com/in/jhon-alberto-betancourt-gonzalez-345557129

Description: Project 3 - Udacity - self driving cars Nanodegree
    (Deep Learning) Build a Traffic Sign Recognition Classifier

Tested on: 
    python 2.7 (3.X should work)
    OpenCV 3.0.0 (3.X or 4.X should work)
    UBUNTU 16.04
"""

# =============================================================================
# LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPEN
# =============================================================================
#importing useful packages
import numpy as np
import cv2
import csv
import os

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# =============================================================================
def duplicates(lst, item):   

    """ returns a list with indexes where element 'item' is repeated in list 'lst'
    Args:
        lst: `list` list to look for repeated elements  
        item: `undefined` elements to look indexes where is repeated in list 'lst'
    Returns: list with indexes where the element is repeated in input list
    """

    return [i for i, x in enumerate(lst) if x == item]

def plot_dataset(means_training, means_validation, means_test, n_classes, save_name = None):

    """ plots dataset data distribution
    Args:
        means_training: `list` list with the amount of training samples for each class 
        means_validation: `list` list with the amount of validation samples for each class
        means_test: `list` list with the amount of testing samples for each class
    Returns:
    """
    
    # Plot parameters
    plt.rcParams["figure.figsize"] = (20,5)
    fig, ax = plt.subplots()
    index = np.arange(n_classes)
    bar_width = 0.2

    # Plot training, validation and test datasets
    rects1 = ax.bar(index, means_training, bar_width, alpha=0.4, color='b', label='Training')
    rects2 = ax.bar(index + bar_width, means_validation, bar_width, alpha=0.4, color='r', label='Validation')
    rects3 = ax.bar(index + 2*bar_width, means_test, bar_width, alpha=0.4, color='g', label='Test')

    # Plotting the graph
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Description')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(map(str, range(n_classes + 1)))
    ax.legend()
    fig.tight_layout()
    
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    
def summary_data_sets(y_train, y_valid, y_test, csv_name):

    """ Returns a list of dictionaries with datasets information
    Args:
        y_train: `list` labels of training dataset
        y_valid: `list` labels of validations dataset
        y_test: `list` labels of testing dataset
    Returns:
        classes_dics: `list` dictionaries with datasets information
    """

    classes_dics = []

    # Get from csv file number of classes and descriptions
    with open(csv_name) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for idx, row in enumerate(csv_reader):

            if idx > 0:

                # Extract index with samples belonging to idx class
                train_idx = duplicates(list(y_train), int(row[0]))
                valid_idx = duplicates(list(y_valid), int(row[0]))
                test_idx = duplicates(list(y_test), int(row[0]))

                # Append new information for class idx
                classes_dics.append({
                    "id": int(row[0]), 
                    "description": row[1], 
                    "train": len(train_idx), 
                    "vali": len(valid_idx), 
                    "test": len(test_idx), 
                    "train_idx": train_idx, 
                    "vali_idx": valid_idx,
                    "test_idx": test_idx})
                
    return classes_dics

def show_dataset(data, classes_dics, data_label, save_name = None):
    
    """ Shows a random sample for each class
    Args:
        data: `np.darray` dataset samples
        classes_dics: `list` dictionaries with datasets information
        data_label: `string` dataset base name
        save_name: `string` absolute path to save plot
    Returns:
    """
    
    columns = 6
    plt.figure(figsize=(20,30))
    n_classes = len(classes_dics)
    
    for idx in range(0, n_classes):
        len_idx = len(classes_dics[idx][data_label+"_idx"])
        idx_data = classes_dics[idx][data_label+"_idx"][np.random.randint(len_idx)]
        ax = plt.subplot(np.ceil(n_classes/columns), columns, idx + 1)
        plt.imshow(data[idx_data])
        ax.set_xlabel(str(idx) + ": " + classes_dics[idx]["description"][0:20])
    
    if save_name is not None:
        plt.savefig(save_name)
    
    plt.show()
    
def gray(img):
    
    """ Converts a image to gray scale
    Args:
        img: `np.darray` image to convert to gray
    Returns:
        dst: `np.darray` image converted to gray scale
    """
    
    # if img.dtype == np.uint8:
    #     img = np.array(img/255.0, dtype=np.float32)
    
    dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst = dst.reshape(32, 32, 1)
    
    return dst

def norm(img):

    """ Normalizes a image with mean zero
    Args:
        img: `np.darray` image to convert to gray
    Returns:
        dst: `np.darray` image normalized
    """

    if img.dtype == np.uint8:
        dst = np.array((img-128.)/128., dtype=np.float32)

    return dst

def add_noisy(img, *argv):
    
    """ Add gaussian noise to image
    Args:
        img" `np.darray` image add Gaussian noisy
    Returns:
        noisy" `np.darray` image with Gaussian noisy
    """
    
    row,col,ch= img.shape
    mean = argv[0]
    var = argv[1]
    sigma = var**argv[2]
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    noisy = np.clip(noisy, a_min = 0, a_max = 255)
    noisy = np.uint8(noisy)

    return noisy

def rot_pers_transform(img, ang=20, d_offset=4):

    """ Rotate and move image
    Args:
        img" `np.darray` image to apply rotations and displacement transformation
        ang" `int` +- value to generate random angle to rotate
        d_offset" `int` +- value to generate random displacement value
    Returns:
        dst" `np.darray` image with spacial transformations
    """

    rows, cols, _ = img.shape

    # Find rotation matrix
    angle = np.random.randint(-ang, ang)
    M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)), angle, 1)

    # Apply x and y axis displacements
    x_offset = np.random.randint(-d_offset, d_offset)
    y_offset = np.random.randint(-d_offset, d_offset)
    M[0][2] += x_offset
    M[1][2] += y_offset

    dst = cv2.warpAffine(img, M, (cols, rows)).astype(int)
    dst = np.uint8(dst)

    return dst

def balance_data(data, labels, classes_dics, data_label = 'train', desired_samples = 1000):
    
    """ Complete and balance dataset
    Args:
       data: `np.darray` dataset samples
       labels: `np.darray` samples classes index
       classes_dics: `list` dictionaries with datasets information
       data_label: `string` base dataset name
       desired_samples: `int` number of desired samples per class
    Returns:
    """

    new_data = []; new_labels = []
    for idx, dic in enumerate(classes_dics):
        
        if dic[data_label] < desired_samples:
            diff = desired_samples - dic[data_label]
            
            for num in range(diff):
                rand_idx = np.random.randint(len(dic[data_label+'_idx']))
                sample = data[dic[data_label+'_idx'][rand_idx]]
                
                # Apply transformations
                sample = add_noisy(sample, 20, np.random.randint(150), 0.5)
                sample = rot_pers_transform(sample)
                
                # Append new values to lists
                new_labels.append(idx)
                new_data.append(sample)
             
    # Concatenate new data
    labels = np.concatenate((labels, new_labels))
    data = np.concatenate((data, new_data))

    return data, labels
        
def get_img_transformations(img):
    
    """ Applies spacial and distortion transformations to an image
    Args:
        img" `np.darray` image to apply transformation
    Returns:
    """

    # Apply transformations
    X_valid_noise = add_noisy(img, 20, np.random.randint(150), 0.5)
    X_valid_rota  = rot_pers_transform(X_valid_noise)
    X_valid_gray  = gray(X_valid_rota)

    # Plot results
    _, vframes = plt.subplots(nrows=1, ncols=4)
    vframes[0].imshow(img)
    vframes[1].imshow(X_valid_noise)
    vframes[2].imshow(X_valid_rota)
    vframes[3].imshow(X_valid_gray.squeeze(), cmap='gray')

    idx = 0
    file_name = "writeup_files/data_augmentation_sample_{}.png".format(idx)
    while os.path.isfile(file_name):
        idx += 1
        file_name = "writeup_files/data_augmentation_sample_{}.png".format(idx)

    # Save figure
    plt.savefig(file_name)

def plot_history(history, save_name=None):

    A4_PORTRAIT = (8.27, 11.69)
    A4_LANDSCAPE = A4_PORTRAIT[::-1]

    plt.figure(figsize=A4_LANDSCAPE)
    plt.title('Model training history: {}'.format("model_training"))
    plt.plot(history)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    if save_name is not None:
        plt.savefig(save_name)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, figsize=(300,40)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# =============================================================================
# W = 14 # input layer has a width
# H = 14 # input layer has a height
# F = 5 # convolutional layer has a filter size 
# P = 0 # padding 
# S = 1 # stride
# K = 16 # number of filters
# W_out =((W-F+2*P)/S) + 1 # width of the next layer
# H_out =((H-F+2*P)/S) + 1 # heihgt of the next layer
# D_out = K # the output depth would be equal to the number of filters

# print("W_out:", W_out)
# print("H_out:", H_out)
# print("D_out:", D_out)