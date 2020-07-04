# import the necessary packages
from imutils.video import FPS
import numpy as np
# import argparse
import imutils
import dlib
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
#   help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
#   help="path to Caffe pre-trained model")
# ap.add_argument("-v", "--video", required=False,
#   help="path to input video file")
# ap.add_argument("-l", "--label", required=True,
#   help="class label we are interested in detecting + tracking")
# ap.add_argument("-o", "--output", type=str,
#   help="path to optional output video file")
# ap.add_argument("-c", "--confidence", type=float, default=0.2,
#   help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
args = {}
args['prototxt']   = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
args['model']      = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
args['video']      = 'input/street.mp4'
args['label']      = 'person'
args['output']     = 'output/street_output.avi'
args['confidence'] =  0.2




# python track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#   --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video input/street.mp4 \
#   --label person --output output/street_output.avi


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
  "sofa", "train", "tvmonitor"]
# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# initialize the video stream, dlib correlation tracker, output video
# writer, and predicted class label
print("[INFO] starting video stream...")
# vs = cv2.VideoCapture(args["video"])
vs = cv2.VideoCapture(0)

tracker = None
writer = None
label = ""
# start the frames per second throughput estimator
fps = FPS().start()

# Updating Tracker every update_every frame
i = -1
update_every = 50

# loop over frames from the video file stream
while True:
  # grab the next frame from the video file
  (grabbed, frame) = vs.read()
  # check to see if we have reached the end of the video file
  if frame is None:
    break
  # resize the frame for faster processing and then convert the
  # frame from BGR to RGB ordering (dlib needs RGB ordering)
  frame = imutils.resize(frame, width=600)
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # if we are supposed to be writing a video to disk, initialize
  # the writer
  if args["output"] is not None and writer is None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 30,
      (frame.shape[1], frame.shape[0]), True)


  # if our correlation object tracker is None we first need to
  # apply an object detector to seed the tracker with something
  # to actually track
  i += 1
  if i % update_every == 0 :
    print('Updating!')
    tracker = None
  if tracker is None:
    print('No tracker..')
    # grab the frame dimensions and convert the frame to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
    # pass the blob through the network and obtain the detections
    # and predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one detection is made
    if len(detections) > 0:
      # find the index of the detection with the largest
      # probability -- out of convenience we are only going
      # to track the first object we find with the largest
      # probability; future examples will demonstrate how to
      # detect and extract *specific* objects
      i = np.argmax(detections[0, 0, :, 2])
      # grab the probability associated with the object along
      # with its class label
      conf = detections[0, 0, i, 2]
      label = CLASSES[int(detections[0, 0, i, 1])]


      # filter out weak detections by requiring a minimum
      # confidence
      if conf > args["confidence"] and label == args["label"]:
        # compute the (x, y)-coordinates of the bounding box
        # for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        print(box)

        (startX, startY, endX, endY) = box.astype("int")
        # construct a dlib rectangle object from the bounding
        # box coordinates and then start the dlib correlation
        # tracker
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb, rect)
        # draw the bounding box and text for the object
        cv2.rectangle(frame, (startX, startY), (endX, endY),
          (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 15),
          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


  # otherwise, we've already performed detection so let's track
  # the object
  else:
    # update the tracker and grab the position of the tracked
    # object
    tracker.update(rgb)
    pos = tracker.get_position()
    # unpack the position object
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    centre_coords = ((startX+endX)//2, (startY+endY)//2)
    # draw the bounding box from the correlation object tracker
    cv2.circle(frame, centre_coords, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color=(0, 255, 0), thickness=2)
    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    print(centre_coords)

  # check to see if we should write the frame to disk
  if writer is not None:
    writer.write(frame)
  # show the output frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break
  # update the FPS counter
  fps.update()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# check to see if we need to release the video writer pointer
if writer is not None:
  writer.release()
# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()


