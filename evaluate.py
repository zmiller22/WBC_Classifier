# USAGE
# python evaluate.py --show_data 1
# OR
# python evaluate.py --show_data 1 --test_data dataset2-master/images/TEST --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle

# Import the necessary packages
import matplotlib
matplotlib.use("Agg")

from keras.models import load_model
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import random
import sys
import cv2
import os

# Read in any command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--show_data", required = True,
	help="list all data classifications, 1 for yes 0 for no")
ap.add_argument("-t", "--test_data", required=False,
	help="path to folder containing testing classification_images")
ap.add_argument("-m", "--model", required=False,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=False,
	help="path to output label binarizer")

args = vars(ap.parse_args())

# If user enters paths via command line, use those
if len(sys.argv) == 9:
	SHOW_PREDICTIONS = args['show_data']
	TEST_DATA_PATH = args['test_data']
	MODEL_PATH = args['model']
	LABEL_BIN_PATH = args['label_bin']

# Otherwise use the default file paths for this file structure
else:
	SHOW_PREDICTIONS = args['show_data']
	TEST_DATA_PATH = 'dataset2-master/images/TEST'
	MODEL_PATH = 'output/smallvggnet.model'
	LABEL_BIN_PATH = 'output/smallvggnet_lb.pickle'

# Initialize the testing data and label arrays
print("Loading images...")
test_data = []
test_data_simple = []
test_labels = []
test_labels_simple = []

# Load testing data
test_imagePaths = sorted(list(paths.list_images(TEST_DATA_PATH)))
random.seed(42)
random.shuffle(test_imagePaths)

for imagePath in test_imagePaths:
# We need to create two test_data arrays because two functions that
# are used later will expect different input shapes. test_data_simple
# will be a 3 element vector with the image, while test_data will be
# a 4 element vector
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	test_data_simple.append(image)
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))
	test_data.append(image)

# Extract the class label from the image path and update the
# test_labels list
	label = imagePath.split(os.path.sep)[-2]
	test_labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
test_data = np.array(test_data, dtype="float") / 255.0
test_data_simple = np.array(test_data_simple, dtype="float") / 255.0
test_labels = np.array(test_labels)
test_labels_simple = test_labels

# Convert the test_labels into vectors
lb = LabelBinarizer()
test_labels = lb.fit_transform(test_labels)

# Load model
model = load_model(MODEL_PATH)

# Print each predicted label, actual label, and confidence if the
# user asks for the information
print("Displaying predictions...")
if int(SHOW_PREDICTIONS) == 1:
	for i in range(len(test_data)):

		preds = model.predict(test_data[i])
		max_prob = preds.argmax(axis=1)[0]
		label_pred = lb.classes_[max_prob]
		label_actual = test_labels_simple[i]
		confidence = preds[0][max_prob]*100

		print("Predicted:", label_pred, " Actual:", label_actual,
			  "Confidence:", confidence)

# Print overall information
print("Creating overall report...")
predictions = model.predict(test_data_simple, batch_size=32)
print(classification_report(test_labels.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

