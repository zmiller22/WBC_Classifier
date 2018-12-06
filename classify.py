# USAGE
# python classify.py --image dataset2-master/images/TEST
# OR
# python classify.py --image dataset2-master/images/TEST --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle

# Import the necessary packages
from keras.models import load_model
#from train_vgg import MODEL_PATH, LABEL_BIN_PATH, BEST_MODEL_PATH
from imutils import paths
import os
import argparse
import pickle
import cv2
import sys

# Get path to the image or folder of classification_images to be tested and
# save it as IMAGE_PATH
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=False,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=False,
	help="path to output label binarizer")

args = vars(ap.parse_args())

# If user enters paths via command line, use those
if len(sys.argv) == 7:
	IMAGE_PATH = args['image']
	MODEL_PATH = args['model']
	LABEL_BIN_PATH = args['label_bin']

# Otherwise allow them to enter constants into the code
else:
	IMAGE_PATH = args['image']
	MODEL_PATH = 'output/smallvggnet.model'
	LABEL_BIN_PATH = 'output/smallvggnet_lb.pickle'

# Load the model and binarizer
model = load_model(MODEL_PATH)
lb = pickle.loads(open(LABEL_BIN_PATH, "rb").read())

# Create a list of all the image paths and loop over it
imagePaths = sorted(list(paths.list_images(IMAGE_PATH)))
print("Classifying classification_images...")

for imagePath in imagePaths:
	# Load and format image
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	image = image.astype("float") / 255.0
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

	# Find the highest probability classification for the image
	preds = model.predict(image)
	max_prob = preds.argmax(axis=1)[0]

	label_pred = lb.classes_[max_prob]
	confidence = preds[0][max_prob] * 100
	name = imagePath.rstrip(os.sep)
	name = os.path.basename(name)

	# Print the result
	print("Classified", name, "as", label_pred, "with", confidence, "percent confidence")
