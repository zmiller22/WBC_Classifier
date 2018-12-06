# USAGE
# python train_vgg.py
# OR
# python train_vgg.py --train_data dataset2-master/classification_images/TRAIN  --test_data dataset2-master/classification_images/TEST --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png

# Import the necessary packages
import matplotlib
matplotlib.use("Agg")

from smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import callbacks
from keras.models import load_model
from keras import optimizers
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import random
import pickle
import cv2
import os

# Read in command line arguments (paths to any necessary files)
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train_data", required=False,
	help="path to folder containing training classification_images")
ap.add_argument("-t", "--test_data", required=False,
	help="path to folder containing testing classification_images")
ap.add_argument("-m", "--model", required=False,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=False,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=False,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# If user enters paths via command line, use those
if len(sys.argv) == 11:
	TRAIN_DATA_PATH = args['train_data']
	TEST_DATA_PATH = args['test_data']
	MODEL_PATH = args['model']
	LABEL_BIN_PATH = args['label_bin']
	PLOT_PATH = args['plot']

# Otherwise use default paths for this file structure
else:
	TRAIN_DATA_PATH = 'dataset2-master/images/TRAIN'
	TEST_DATA_PATH = 'dataset2-master/images/TEST'
	MODEL_PATH = 'output/smallvggnet.model'
	LABEL_BIN_PATH = 'output/smallvggnet_lb.pickle'
	PLOT_PATH = 'output/smallvggnet_plot.png'

# Initialize the training/testing data and label arrays
print("[INFO] loading images...")
train_data = []
train_labels = []
test_data = []
test_labels = []

# Load training data and shuffle it using code adapted from Adrian Rosebrock
train_imagePaths = sorted(list(paths.list_images(TRAIN_DATA_PATH)))
random.seed(42)
random.shuffle(train_imagePaths)

# Loop over the training classification_images
for imagePath in train_imagePaths:
	# Load, resize, and save the image to the training_data array
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	train_data.append(image)

	# Get the label of the image and save it to the training_labels array
	label = imagePath.split(os.path.sep)[-2]
	train_labels.append(label)

# Scale the pixel intensities to the range [0, 1]
train_data = np.array(train_data, dtype="float") / 255.0
train_labels = np.array(train_labels)

# Load testing classification_images by repeating the same proccess as above, but saving
# to the test_data and test_labels arrays
test_imagePaths = sorted(list(paths.list_images(TEST_DATA_PATH)))
random.seed(42)
random.shuffle(test_imagePaths)

for imagePath in test_imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	test_data.append(image)

	label = imagePath.split(os.path.sep)[-2]
	test_labels.append(label)

test_data = np.array(test_data, dtype="float") / 255.0
test_labels = np.array(test_labels)

# This binarizes the test labels and training labels and also transforms
# them into vectors so we can access the labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Initialize the neural network from smallvggnet.py
model = SmallVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

# List out constants for the CNN. These are optimized for the blood cell classification_images,
# so you may want to change these for different classification_images or datasets
INIT_LR = 0.001 # learning rate
EPOCHS = 60 # number of epochs
BS = 32 # batch size
OPT = optimizers.SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True) # optimizing function

print("Training CNN...")
# Initialize model
model.compile(loss="categorical_crossentropy", optimizer=OPT, # !!! for 2 class classification, use binary_crossentropy
	metrics=["accuracy"])

# Monitor the progress of the CNN accuracy and save the most accurate model
# to BEST_MODEL_PATH
saveBestModel = callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_acc', verbose=1, save_best_only=True,
	save_weights_only=False, mode='auto')

# Train the model
H = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
    epochs=EPOCHS, batch_size=32, callbacks=[saveBestModel])

# Evaluate the network and print a simple summary of the results
print("Evaluating network...")

model = load_model(MODEL_PATH)
predictions = model.predict(test_data, batch_size=32)
print(classification_report(test_labels.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# Plot the training loss and accuracy. This code is adapted from Adrian Rosebrock
# and prints a standard graph to show the development of the model
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(PLOT_PATH)

# Save the model and label_binarizer to the specified paths on
# disk so we can use it later
print("Saving model and labels to disk...")
model.save(MODEL_PATH)
f = open(LABEL_BIN_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()