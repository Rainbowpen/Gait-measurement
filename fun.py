import numpy as np
import math
import glob
import os
import six.moves.urllib as urllib 
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import cv2
import threading

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

# Import the object detection module.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

		
# Patches:
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

class config:

	#PATH_TO_LABELS = '/home/simon/Documents/project/blue_point/models/research/object_detection/data/mscoco_label_map.pbtxt''
	PATH_TO_LABELS = '/home/simon/Documents/project/blue_point/my_label_map.pbtxt'
	VIDEO_NUMBER = '19'
	TEST_VIDEO_PATH = './test_video/foot_' + VIDEO_NUMBER + '.mp4'
	CAMERA = 0
	URL = 'http://localhost:8081'
	#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'#
	MODEL_NAME = 'blue_point'
	VIDEO_SAVE_PATH = '/home/simon/Videos/video_output_' + VIDEO_NUMBER + '.mp4'
	RULER = 0.45
	CM = 40



class run:

	category_index = label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)



	def load_model(model_name):
		# Loader
		model_dir = './tf_detection_model_zoo/' + str(model_name)
		model_dir = pathlib.Path(model_dir)/"saved_model"
		model = tf.saved_model.load(str(model_dir))
		model = model.signatures['serving_default']
		return model


	
	def run_inference_for_single_image(model, image):
		# Add a wrapper function to call the model, and cleanup the outputs:
		
		image = np.asarray(image)
		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(image)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis,...]

		# Run inference
		output_dict = model(input_tensor)
		# All outputs are batches tensors.
		# Convert to numpy arrays, and take index [0] to remove the batch dimension.
		# We're only interested in the first num_detections.
		num_detections = int(output_dict.pop('num_detections'))
		output_dict = {key:value[0, :num_detections].numpy() 
								for key,value in output_dict.items()}
		output_dict['num_detections'] = num_detections
		
		# detection_classes should be ints.
		output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

		# Handle models with masks:
		if 'detection_masks' in output_dict:
		# Reframe the the bbox mask to the image size.
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					output_dict['detection_masks'], output_dict['detection_boxes'],
								image.shape[0], image.shape[1])      
			detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
			tf.uint8)
			output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

		return output_dict


	# Run it on each test image and show the results:

	def inference(model, image):
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = np.array(image)
		# Actual detection.
		output_dict = run.run_inference_for_single_image(model, image_np)
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			run.category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)
		
		# This is the way I'm getting my coordinates
		boxes = output_dict['detection_boxes']
		#lable = output_dict['detection_classes']
		# get all boxes from an array
		max_boxes_to_draw = boxes.shape[0]
		max_boxes_to_draw
		# get scores to get a threshold
		scores = output_dict['detection_scores']
		# this is set as a default but feel free to adjust it to your needs
		min_score_thresh=.5
		find_object = []
		# iterate over all objects found
		#print('------------------------------------------')
		for i in range(min(max_boxes_to_draw, boxes.shape[0])):
			if scores is None or scores[i] > min_score_thresh:
				# boxes[i] is the box which will be drawn
				#class_name = self.category_index[output_dict['detection_classes'][i]]['name']
				#print (class_name, boxes[i])
				ymin = boxes[i][0]
				xmin = boxes[i][1]
				ymax = boxes[i][2]
				xmax = boxes[i][3]
				find_object.append([xmin, ymin, xmax, ymax])
		
		return image_np, find_object

	def check_is_two_foot(find_object):
	# if have two foot, return feet data
		if len(find_object) == 2:
			iou = run.bb_intersection_over_union(find_object[0], find_object[1])
			iou_foot_1 = run.bb_intersection_over_union([0,0,1,1], find_object[0])
			iou_foot_2 = run.bb_intersection_over_union([0,0,1,1], find_object[1])
			if iou > 0.55 or iou_foot_1 > 0.4 or iou_foot_2 > 0.4:
				return None

			if find_object[0][0] <= find_object[1][0]:
				left_foot = find_object[1]
				right_foot = find_object[0]
			elif find_object[0][0] > find_object[1][0]:
				left_foot = find_object[0]
				right_foot = find_object[1]
			#lfoot = left_foot
			#rfoot = right_foot
			feet = [left_foot, right_foot]
		else:
			feet = None


		return feet


	def bb_intersection_over_union(boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# return the intersection over union value
		return iou


	def write_to_video(path, image_array, size):
		video_output = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (size))
		cv2.imwrite('./preview.jpg', image_array[-1])
		for i in range(len(image_array)):
			video_output.write(image_array[i])
		video_output.release()




class ipcamCapture:
	# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
		
	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()
