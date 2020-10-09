import imutils
import cv2

def preprocess(image, width, height):
    # width, height are target dims
    # dim of input image
    h, w = image.shape[:2]

    #resize along longest side
    # doing this here to keep aspect ratio
    if w > h:
        image = imutils.resize(image, width=width)

    else:
        image = imutils.resize(image, height=height)

    # pad along other dimension
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad and resize to account for any rounding errors
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image
