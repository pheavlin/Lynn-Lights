from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np;
import socket
from util.transforms import four_point_transform
from util.draw import draw_image_in_window

UDP_IP = "169.254.183.163"
UDP_PORT = 9850
MESSAGE = b"a\r"

# Adapted from https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html, https://github.com/opencv/opencv/tree/3.4/samples/python/tutorial_code/video/background_subtraction/bg_sub.py

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.mov')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

# Setup Background Subtractor
numFramesToIncludeInBackgroundModel = 30 * 60 # 1minute at 30fps
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(numFramesToIncludeInBackgroundModel, 400.0, False)
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Setup Blob Detector
# ref: https://stackoverflow.com/a/58644329, https://learnopencv.com/blob-detection-using-opencv-python-c/, http://amroamroamro.github.io/mexopencv/opencv/detect_blob_demo.html
params = cv.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 2000
params.filterByColor = True
params.blobColor = 255
params.minArea = 100
params.maxArea = 3000
params.filterByArea = True
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv.SimpleBlobDetector_create(params)

# Load Video Sample
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
# TODO replace with live video Capture

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Correct Camera w/ 4 point transform
    pts = np.array([
        [frame.shape[1]*0.45,0], [frame.shape[1]*0.75,0],
        [frame.shape[1],frame.shape[0]], [0,frame.shape[0]]
    ], dtype = "float32")
    frameCorrected_raw = four_point_transform(frame, pts)
    # crop out partial ground tiles
    frameCorrected = frameCorrected_raw[0:frameCorrected_raw.shape[0], 32:frameCorrected_raw.shape[1]]

    for pt in pts:
        cv.circle(frame, (int(pt[0]), int(pt[1])), 5, (255,0,0), -1)

    # Perform Background Substraction
    fgMask = backSub.apply(frameCorrected)

    # Detect Blobs
    keypoints = detector.detect(fgMask)
    
    # Generate image with detected blobs as red circles
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    blank_image = np.zeros((frameCorrected.shape[0],frameCorrected.shape[1],3), np.uint8)
    blobIm = cv.drawKeypoints(blank_image, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    blobPlusFrameIm = cv.drawKeypoints(frameCorrected, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display Frame Number
    # get the frame number and write it on the current frame
    # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # Detect Vertical Slices (corresponding with coffers overhead)
    numberOfRegions = 6
    regionRange = range(0, numberOfRegions)
    regionWidth = int(frameCorrected.shape[1] / numberOfRegions)
    triggeredRegions = []
    regionBounds = []
    for regionIdx in regionRange:
        triggeredRegions.append(False)
        regionBounds.append([(regionIdx * regionWidth, 0), (regionIdx * regionWidth + regionWidth, frameCorrected.shape[0])])

    for keypoint in keypoints:
        for idx, region in enumerate(regionBounds):
            radius = keypoint.size/2
            if keypoint.pt > tuple(r-radius for r in region[0]) and keypoint.pt < tuple(r+radius for r in region[1]):
                triggeredRegions[idx] = True

    for idx, region in enumerate(regionBounds):
        if triggeredRegions[idx] == True:
            MESSAGE = b"a\r"
            sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
            sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

    triggering = np.zeros((frameCorrected.shape[0],frameCorrected.shape[1],3), np.uint8)
    for idx, region in enumerate(regionBounds):
        cv.rectangle(triggering, region[0], region[1], (255,255,255), (1, -1)[triggeredRegions[idx]])

    # Show videos
    x = 0
    y = 0
    draw_image_in_window('Video', frame, x, y)
    x += frame.shape[1] + 10

    draw_image_in_window('Video Corrected', frameCorrected, x, y)
    x += frameCorrected.shape[1] + 10

    draw_image_in_window('Foreground', fgMask, x, y)
    x = 0
    y += frameCorrected.shape[0] + 40

    draw_image_in_window("Blobs", blobIm, x, y)
    x += blobIm.shape[1] + 10

    draw_image_in_window("Blobs & Video", blobPlusFrameIm, x, y)
    x += blobPlusFrameIm.shape[1] + 10

    draw_image_in_window("Slice Triggering", triggering, x, y)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
