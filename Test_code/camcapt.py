#!/usr/bin/env python3
import os
import cv2
import rospy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from tensorflow.keras.models import load_model
from cv_bridge import CvBridge, CvBridgeError



bridge = CvBridge()
#model = load_model(os.path.join(os.path.dirname(__file__), "shape_classifier_le_net_5.h5"))
#graph = tf.get_default_graph()

def image_cb(data):	
	cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

	image = preprocess_image(cv_image)

	#global graph
	#with graph.as_default():
	#	prediction = classes[np.squeeze(np.argmax(model.predict(image), axis=1))]
	#	move(prediction)
		
def preprocess_image(image):
	resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
	return np.resize(resized_image, (1, IMAGE_WIDTH, IMAGE_WIDTH, 3))
