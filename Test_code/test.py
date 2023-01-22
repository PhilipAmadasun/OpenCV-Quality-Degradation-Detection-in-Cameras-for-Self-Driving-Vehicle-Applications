import tensorflow as tf
import cv2
import keras
from keras.models import load_model
import numpy as np
size=224
model = tf.keras.models.load_model('/home/philip/QAME696/savedmodel/CNNModel')
image = cv2.imread("/home/philip/QAME696/rain.png")
resized=cv2.resize(image,(size,size)) # keras.util.load_image
prediction=model.predict(resized[np.newaxis])
# Check its architecture
prediction