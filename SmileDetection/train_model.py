from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from models.nn.conv import LeNet
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", help="path to save models weights", required=True)
ap.add_argument("-d", "--dataset", help="path to input dataset of faces", required=True)
args = vars(ap.parse_args())

data = []
labels = []

# loop over input images
for imagePath in paths.list_images(args["dataset"]):
    # load image, preprocess, and store
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, 28)
    image = img_to_array(image)
    data.append(image)
    # get class label from file directory
    # SMILEs/positives/positives7/10007.jpg
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# normalize
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# encode to vector (one-hot)
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

# Data imbalance handling
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition train test
# This stratify parameter makes a split so that the proportion
# of values in the sample produced will be the same as the proportion
# of values provided to parameter stratify.
trainX, testX, trainY, testY = train_test_split(data,
                                                labels, test_size=0.2,
                                                stratify=labels,
                                                random_state=42)

# Initialize models
print("[INFO] compiling models...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeight,
              batch_size=64, epochs=15, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=le.classes_))

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

