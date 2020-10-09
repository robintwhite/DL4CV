import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors = None):
        # store image preprocessors
        self.preprocessors = preprocessors

        # if preprocessors are None, initialize them
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose = 1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract class label
            # assuming path has format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check preprocessors
            if self.preprocessors is not None:
                # loop over the preprocessors and apply to each image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            # show an update for verbose
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                                                      len(imagePaths)))

        return np.array(data), np.array(labels)