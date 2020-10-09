from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from pygmy.nn.conv import LeNet
from pygmy.utils.captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", help="path to save models weights", required=True)
ap.add_argument("-d", "--dataset", help="path to input dataset", required=True)
args = vars(ap.parse_args())

data = []
labels = []

# loop over input images
for imagePath in paths.list_images(args["dataset"]):
    # load image, preprocess, and store
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)
    # get class label from file directory
    # root_directory/class_label/image_filename.jpg
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# normalize
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# train test split 75% train, 25% test
trainX, testX, trainY, testY = train_test_split(data, labels,
                                                test_size=0.25,
                                                random_state=42)

# encode to vector (one-hot)
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

# Initialize models
print("[INFO] compiling models...")
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=15, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))
print("[INFO] serializing network...")
model.save(args["models"])

# plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

