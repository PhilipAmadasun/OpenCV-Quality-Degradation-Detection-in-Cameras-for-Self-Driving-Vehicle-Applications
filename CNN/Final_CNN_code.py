import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import cv2
import os

import numpy as np

#######################################################################


def get_data(data_dir,labels,img_size):
    '''(str, list, int) -> numpy array
    
    Reads images from appropriate folders, resize them and assigns labels.
    Returns a numpy array where each element is [224x224 img, label]
    '''
    
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)    # Find subfolder e.g. traindata\rainy
        classLabel = labels.index(label)         # Assign discrete number to each label
        
        for img in os.listdir(path):
            
            imgInput = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
            resizedImg = cv2.resize(imgInput, (img_size, img_size)) # Reshaping images to preferred size
            data.append([resizedImg, classLabel])
            
    return np.array(data)


def prepare_data(trainingpath,testingpath,labels,img_size):
    '''(str, str, list, int) -> np.array, np.array, np.array, np.array
    
    Reads images using get_data() function and prepare training and testing 
    inputs and outputs.
    
    Returns numpy arrays of training and testing images, and their labels.
    '''
    
    trainingdata = get_data(trainingpath,labels,img_size)
    testingdata = get_data(testingpath,labels,img_size)
    
    # Prepare Training and Testing input and output data
    # Input := Image
    # Output := Label
    x_train, x_test, y_train, y_test = [],[],[],[]
    
    for image, label in trainingdata:
        x_train.append(image)
        y_train.append(label)

    for image, label in testingdata:
        x_test.append(image)
        y_test.append(label)
    
    # Normalize the images between ranges 0-1
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    
    # Reshaping the image arrays to 224x224 for use with Keras
    x_train.reshape(-1, img_size, img_size, 1)
    x_test.reshape(-1, img_size, img_size, 1)
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Data Augmentation to produce variations in the training data
    data_augmentation = ImageDataGenerator(
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True)  # randomly flip images

    data_augmentation.fit(x_train)
    
    
    return x_train, x_test, y_train, y_test
    

def define_model():
    '''(None) -> Keras Model
    
    Defines the CNN model architecture using Keras library and returns the
    model.
    '''
    
    model = Sequential()
    
    # First Convolutional Layer with 32 kernels of 3x3 size
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
    model.add(MaxPool2D())
    
    # Second Convolutional Layer with 64 kernels of 3x3 size
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    # Third Convolutional Layer with 128 kernels of 3x3 size
    model.add(Conv2D(128, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    
    # Dropout layer with 30% drop rate
    model.add(Dropout(0.3))
    
    # Convert 2D image tensors to vector tensor for fully-connected layers
    model.add(Flatten())
    
    # Two fully-connected layers with 1000 neurons each
    model.add(Dense(1000,activation="relu"))
    model.add(Dense(1000,activation="relu"))
    
    # Output layer of classifier with 4 neurons each corresponding to a class
    model.add(Dense(4, activation="softmax"))

    # SGD optimization algorithm Adam
    opt = Adam(learning_rate=0.000001)
    
    model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
    
    return model

def run_training(model,x_train,x_test,y_train,y_test,epochs):
    '''(Keras model, np.array, np.array, np.array, np.array, int) -> Keras history obj
    
    Runs the training on the defined Keras neural network using the training
    and testing data. 
    
    Returns the training progress report. 
    '''
    
    trainingprogress = model.fit(x_train,y_train,epochs = epochs , validation_data = (x_test, y_test))
    
    # Save the trained model
    model.save('savedmodel/CNNModel')
    
    return trainingprogress
    
def plot_training_progress(progress,numepochs):
    '''(Keras history obj, int) -> None
    
    Plots the Training and Testing Accuracy and Loss throughout the training. 
    '''
    
    acc = progress.history['accuracy']
    val_acc = progress.history['val_accuracy']
    loss = progress.history['loss']
    val_loss = progress.history['val_loss']

    epochs_range = range(numepochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Testing Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Testing Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Testing Loss')
    plt.legend(loc='lower left')
    plt.title('Training and Testing Loss')
    plt.show()
    
    
if __name__ == '__main__':
    
    # Initialization variables
    labels = ['rainy','clear','dusty','smudgy,not clear'] #'rainy'
    img_size = 224
    trainingepoch = 100
    
    # Assign image file paths for the training and testing image data
    trainingpath = '~/OpenCV-Quality-Degradation-Detection-in-Cameras-for-Self-Driving-Vehicle-Applications/trainingdata'
    testingpath = '~/OpenCV-Quality-Degradation-Detection-in-Cameras-for-Self-Driving-Vehicle-Applications/testdata'
    
    # Assign labels to data, normalize and prepare the image arrays
    x_train, x_test, y_train, y_test = prepare_data(trainingpath, testingpath, labels, img_size)
    
    # Define the CNN architecture and get it ready to train
    model = define_model()
    
    # Run the model training for number of epochs described
    trainingprogress = run_training(model, x_train, x_test, y_train, y_test, trainingepoch)
    
    # Plot the training/testing accuracy and loss
    plot_training_progress(trainingprogress, trainingepoch)
