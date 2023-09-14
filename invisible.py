import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

count = 0
background = None    # to capture the initial frame

while cap.isOpened():
    ret, img = cap.read()

    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis=1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate masks to detect red color
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([5, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    # Open and dilate the mask image
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverted mask to segment out the red color from the frame
    mask2 = cv2.bitwise_not(mask1)

    # Segment the red color part out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(img, img, mask=mask2)

    if background is None:
        background = img.copy()  # Initialize the background with the first frame

    # Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    finalOp = cv2.addWeighted(res1, 1, res2, 1, 0)

    out.write(finalOp)
    cv2.imshow("boom", finalOp)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
