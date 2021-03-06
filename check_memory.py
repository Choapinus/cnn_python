# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from contextlib import redirect_stdout
from time import sleep

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-n", "--n_images", type=int, required=False, default=1000,
	help="path to output model")
ap.add_argument("-e", "--epochs", type=int, required=False, default=200,
	help="epochs to do")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = args["epochs"] # 25
INIT_LR = 1e-2
BS = 20 # debe ser numero cuadrado sino toma menos imagenes

# initialize the data and labels
print("[INFO] loading " + str(args["n_images"]) + " images...")
data = []
labels = []


# andresin
male_file, female_file = open('./male.txt', 'w'), open('./female.txt', 'w')
male_im, female_im = [], []
all_images = sorted(list(paths.list_images(args["dataset"])))
for image in all_images:
	if image.split(os.path.sep)[-2] == 'male' and len(male_im) < args["n_images"]:
		male_im.append(image)
		male_file.write(image+'\n')
	elif image.split(os.path.sep)[-2] == 'female' and len(female_im) < args["n_images"]:
		female_im.append(image)
		female_file.write(image+'\n')
	else:
		continue

male_file.close()
female_file.close()
imagePaths = male_im + female_im

print("[INFO] done with images")

# end andresin


print("[INFO] creating data and labels")
# grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	# image = cv2.resize(image, (178, 218))
	image = cv2.resize(image, (178//2, 218//2))
	# cv2.imshow("iamge", image)
	# cv2.waitKey(0)
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "female" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print("[INFO] done with data and labels")

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

print("data", data.shape)
print("trainX", trainX.shape)
print("testX", testX.shape)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

print("Before deleting data")

sleep(5)

# del data and labels

del data
del labels

print("After deleting data")

sleep(5)

print("trainX", trainX.shape)
print("testX", testX.shape)