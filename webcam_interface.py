import numpy as np
import os, time
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

SAVED_MODEL_PATH = '/media/parthikb/not OS/projects/object-gaming/object_detection/models/research/fine_tuned_model/'

# path to the frozen graph:
PATH_TO_FROZEN_GRAPH = SAVED_MODEL_PATH + 'frozen_inference_graph.pb'

# path to the label map
PATH_TO_LABEL_MAP = SAVED_MODEL_PATH + 'label_map.pbtxt'

# number of classes 
NUM_CLASSES = 1

# 0 > External Webcam
# 1 > Internal Webcam
cap = cv2.VideoCapture(0)

#reads the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            start = time.time()

            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
                )

            # now we take the sum of the score that are outputed per frame
            # if that score gets over some threshold value (here, 0.8), it
            # indicates that hand is detected!
            detected = np.sum(scores)
            if detected >= 0.8:
                print('Hand Detected!', detected)



        # Display output
            cv2.imshow('Object Gaming', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            TIME_PER_FRAME = time.time() - start
            FPS = 1/TIME_PER_FRAME

            print('FPS :', FPS)