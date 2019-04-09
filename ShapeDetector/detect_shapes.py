# import the necessary packages
from ShapeDetector.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
 
# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
def check(frame):
    image = frame
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    #cv2.imshow("resized", resized) 
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("thresh", thresh) 
    #cv2.imshow("gray", gray) 
    #cv2.imshow("blurred", blurred) 
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    def foo(x,y):
        try:
            return x/y
        except ZeroDivisionError:
            return 0

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int(foo(M["m10"], M["m00"]) * ratio)
        cY = int(foo(M["m01"], M["m00"]) * ratio)
        shape = sd.detect(c)    

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        print(c)
        #if shape == "square" or shape == "rectangle":
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        #    0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)