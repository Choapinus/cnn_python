# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-n", "--n_images", type=int, required=False, default=5000,
	help="number of images per class to load")
ap.add_argument("-s", "--split", type=float, required=False, default=0.2,
	help="percent of images that should be in test")
ap.add_argument("-e", "--epochs", type=int, required=False, default=200,
	help="epochs to do")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = args["epochs"] # 25
INIT_LR = 1e-3
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


# grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(args["dataset"])))

print("[INFO] making data and labels")

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
	labels, test_size=args["split"], random_state=42)

print("[INFO] deleting stuff")

del data, labels, male_im, female_im, all_images, imagePath

print("[INFO] done")

print("trainX", trainX.shape)
print("testX", testX.shape)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
# model = LeNet.build(width=178, height=218, depth=3, classes=2)
model = LeNet.build(width=178//2, height=218//2, depth=3, classes=2)
# model.summary()
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# deberia ser SGD porque converge muy rapido
# en keras Adam tiene otros optimizers

opt = SGD(lr=INIT_LR, momentum=0.9)

model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

tbcallback = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# para tensorflow
# tensorboard --logdir ./graph

# train the network
print("[INFO] training network...")

with open('modelsummary.txt', 'w') as f:
	with redirect_stdout(f):
		model.summary()

model.summary()

# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1) # verbose = 1

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[tbcallback]) # verbose = 1

# H = model.fit(x=trainX, y=trainY, batch_size=BS, 
# 	epochs=EPOCHS, verbose=1,
# 	validation_split=0.15, validation_data=(testX, testY))

# H = model.fit(x=trainX, y=trainY, 
# 	epochs=EPOCHS, verbose=1,
# 	validation_data=(testX, testY), 
# 	steps_per_epoch=len(trainX) // BS)



# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Male/Female")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])