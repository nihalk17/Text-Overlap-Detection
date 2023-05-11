# import the necessary packages
from imutils.object_detection import non_max_suppression
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
args = vars(ap.parse_args())

# set model file for text detection
east = 'frozen_east_text_detection.pb'

# load the input image and grab the image dimensions
image = cv2.imread(args['image'])
orig = image.copy()
(H, W) = image.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320,320)
rW = W / float(newW)
rH = H / float(newH)
# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# load the pre-trained EAST text detector
net = cv2.dnn.readNet(east)
# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]
  # loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < 0.5:    # minimum probability required to inspect a region
			continue
		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]
		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)
		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# draw the bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)

# importing pretrained model weights
#setting model configuration
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#loading image for detection 
#and resizing it
I = cv2.imread(args['image'])
img = cv2.resize(I,(320,320))

#Detecting person using loaded mobilenetssdv3 model
#Plotting the bounding box of detected image
m, n, bbox = model.detect(img, confThreshold = 0.5)
for box in bbox:
  cv2.rectangle(image,box,(255,0,0),1)
cv2.imshow('Processed Image',image)
cv2.waitKey(0)

enlarged_img = cv2.resize(image,(720,720))
cv2.imshow('Enlarged Image',enlarged_img)
cv2.waitKey(0)

#Function to check if there is overlap in text and human
#Return 0=>no overlap, 1=>overlap
def check(l1, r1, l2, r2):
  if (l1[0]>=r2[0] or l2[0]>=r1[0] or l1[1]>=r2[1] or l2[1]>=r1[1]):
    return 0
  else:
    return 1

#Iterating over detected text and human to check for their overlap
count = 0
for (startX_1,startY_1,width_1,height_1) in bbox:
  for (startX_2,startY_2,endX_2,endY_2) in boxes:
    count = count + check((startX_1,startY_1),((startX_1+width_1),(startY_1+height_1)),(startX_2,startY_2),(endX_2,endY_2))
    
if count>0:
    print("Overlap")
else:
    print("No Overlap")
