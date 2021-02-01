#Import the neccesary libraries
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2 

# construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt", help='Path to text network file:')
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel", help='Path to weights')
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
#parser.add_argument("--use-gpu", type=bool, default=True,help="boolean indicating if CUDA GPU should be used")
parser.add_argument("--output", default="./detections/output.avi", type=str, help="Outout path to save file")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
if args.use_gpu:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

frame_no = 0
fps = FPS().start()
print("Processing the video...")
while True:
    # Capture frame-by-frame
    return_value, frame = cap.read()
    if return_value:
        frame_no += 1
    else:
        if frame_no == cap.get(cv2.CAP_PROP_FRAME_COUNT):
          print("Video processing complete")
          break
        raise ValueError("No image! Try with another video format")
    
    if frame_no == 1:
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      writer = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size. 
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    people_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            if class_id == 15:
              people_count += 1
            else:
              continue    #ignore other detections

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 0, 200),2)

            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (0, 0, 200), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            
            if people_count == 0:
                cv2.putText(image, "WARNING!No person in frame!", (5, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 0, 0), 1)
            elif people_count > 1:
                cv2.putText(image, "WARNING!Multiple people in frame!", (5, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 0, 0), 1)
                cv2.putText(image, "People detected = {}".format(people_count), (5, 22),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 0, 0), 1)
            else:
                cv2.putText(image, "People detected = {}".format(people_count), (5, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 200, 0), 1)

    writer.write(frame)
    fps.update()
    #if cv2.waitKey(1) >= 0:  # Break with ESC 
        #break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
