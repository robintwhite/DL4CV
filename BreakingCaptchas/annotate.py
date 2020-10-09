from imutils import paths
import argparse
import cv2
import imutils
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory")
ap.add_argument("-a", "--annot", required=True,
                help="path to root directory of annotations")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
counts = {}

for i, imagePath in enumerate(imagePaths):
    print("[INFO] process image {}/{}".format(i + 1,
                                              len(imagePaths)))

    try:
        # load and convert to greyscale
        # pad image to make sure no parts touch boundary
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8,
                                  cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        #find contours keeping only the 4 largest
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        for c in cnts:
            #compute the bounding box and extract digit
            (x,y,w,h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w +5]
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            if key ==ord("`"):
                print("[INFO] ignoring character")
                continue

            # grab the key that was pressed and construct the path
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            count = counts.get(key, 1) # counts of each digit, 1 if doesn't exist (first time)
            p = os.path.sep.join([dirPath, "{}.png".format(
                str(count).zfill(6))])
            cv2.imwrite(p, roi)

            counts[key] = count + 1

        # leave on ctrl+c. Still requires a key press for active window
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
    # except:
    #     print("[INFO] unknown error has occurred, skipping image...")
    except Exception as ex:
        print(ex)



