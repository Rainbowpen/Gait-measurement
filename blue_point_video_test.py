## Object Detection for find blue point
#
#### depend tensorflow version == 2.x


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


# Loader
def load_model(model_name):
	model_dir = './tf_detection_model_zoo/' + str(model_name)
	model_dir = pathlib.Path(model_dir)/"saved_model"
	model = tf.saved_model.load(str(model_dir))
	model = model.signatures['serving_default']
	return model


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = '/home/simon/Documents/project/blue_point/models/research/object_detection/data/mscoco_label_map.pbtxt'
PATH_TO_LABELS = '/home/simon/Documents/project/blue_point/my/my_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# For the sake of simplicity we will test on 2 images:

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./test_images')
TEST_VIDEO_PATH = './test_video/foot_10.mp4'


# Load an object detection model:

model_name = 'blue_point' #'ssdlite_mobilenet_v2_coco_2018_05_09'#'ssd_mobilenet_v1_coco_2017_11_17'#'faster_rcnn_nas_coco_2018_01_28'#
detection_model = load_model(model_name)


# Check the model's input signature, it expects a batch of 3-color images of type uint8: 
print(detection_model.inputs)

# And retuns several outputs:
print(detection_model.output_dtypes)
print(detection_model.output_shapes)


# Add a wrapper function to call the model, and cleanup the outputs:
def run_inference_for_single_image(model, image):
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

def show_inference(model, image):
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	image_np = np.array(image)
	# Actual detection.
	output_dict = run_inference_for_single_image(model, image_np)
	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		output_dict['detection_boxes'],
		output_dict['detection_classes'],
		output_dict['detection_scores'],
		category_index,
		instance_masks=output_dict.get('detection_masks_reframed', None),
		use_normalized_coordinates=True,
		line_thickness=8)
	
	# This is the way I'm getting my coordinates
	boxes = output_dict['detection_boxes']
	lable = output_dict['detection_classes']
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
			class_name = category_index[output_dict['detection_classes'][i]]['name']
			#print (class_name, boxes[i])
			ymin = boxes[i][0]
			xmin = boxes[i][1]
			ymax = boxes[i][2]
			xmax = boxes[i][3]
			find_object.append([xmin, ymin, xmax, ymax])
	
	if len(find_object) == 2:
		if find_object[0][0] < find_object[1][0]:
			left_foot = find_object[1]
			right_foot = find_object[0]
		elif find_object[0][0] > find_object[1][0]:
			left_foot = find_object[0]
			right_foot = find_object[1]
		lfoot = left_foot
		rfoot = right_foot
		feet = [lfoot, rfoot]
		#height = abs(find_object[0][0] - find_object[1][0])
		#width = abs(find_object[0][1] - find_object[1][1])
		#distance = math.sqrt(pow(height, 2) + pow(width, 2))
		#print('Two foot distance = ' + str(distance))
		#print('==========================================')
	else:
		feet = None

	return image_np, feet



def main():
	# main
	print('>>============================================================================<<')

	video_cap = cv2.VideoCapture(TEST_VIDEO_PATH)
	success, image = video_cap.read()
	image_array = []
	lfoot_data = []
	rfoot_data = []
	
	if success:
		print('Find video.')
	else:
		print('Video not find.')

	while success:
		#print(type(image))
		new_img, data= show_inference(detection_model, image)
		
		if data is not None:
			lfoot_data.append(data[0])
			rfoot_data.append(data[1])
		
		#new_img = cv2.resize(image_np, (960, 540))
		#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
		#cv2.imshow('Color image', new_img)
		#cv2.waitKey(0)
		#print('')
		height = new_img.shape[0]
		width = new_img.shape[1]
		#layers = new_img.shape[2]
		size = (width, height)
		image_array.append(new_img)
		success, image = video_cap.read()

	video_output = cv2.VideoWriter('/home/simon/Videos/video_output_10.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (size))

	for i in range(len(image_array)):
		video_output.write(image_array[i])
	video_output.release()

	lfoot_data_ymax = []
	rfoot_data_ymax = []
	feet_distance = []
	for lf, rf in zip(lfoot_data, rfoot_data):
		lfoot_data_ymax.append(lf[3])
		rfoot_data_ymax.append(rf[3])
		feet_distance.append(abs(lf[3] - rf[3]))

	lfoot_max = max(lfoot_data_ymax)
	lfoot_min = min (lfoot_data_ymax)
	rfoot_max = max(rfoot_data_ymax)
	rfoot_min = min(rfoot_data_ymax)
	lfoot_avg = (lfoot_max - lfoot_min)/2 + lfoot_min
	rfoot_avg = (rfoot_max - rfoot_min)/2 + rfoot_min
	
	def get_step(data_ymax, foot_avg):
	# return step list
		f_box = []
		max_box = []
		min_box = []
		for i in range(0, len(data_ymax)):
			if i == 0:
				continue
			if data_ymax[i] >= foot_avg:
				if min_box != []:
					f_box.append(min(min_box))
					min_box = []
				max_box.append(data_ymax[i])
			if data_ymax[i] < foot_avg:
				if max_box != []:
					f_box.append(max(max_box))
					max_box = []
				min_box.append(data_ymax[i])
		return f_box
	
	lfoot_step = get_step(lfoot_data_ymax, lfoot_avg)
	rfoot_step = get_step(rfoot_data_ymax, rfoot_avg)
	print(lfoot_step)
	print(rfoot_step)


#	front_box = []
#	back_box = []
#	one_step = []
#	fd_box = []
#	for lf, rf in zip(lfoot_data, rfoot_data):
#		front_foot = max(lf[3], rf[3])
#		back_foot = min(lf[3], rf[3])
#		fd_box.append(front_foot - back_foot)
#		if front_foot - back_foot >= 0.2:
#			front_box.append(front_foot)
#			back_box.append(back_foot)
#		else:
#			try:
#				one_step.append([max(front_box), min(back_box)])
#				fd_box = []
#				front_box = []
#				back_box = []
#			except:
#				pass

#	box = []
#	for i in one_step:
#		box.append(i[0] - i[1])
#	feet_dis_max = max(box)
#	feet_dis_min = min(box)


	print('Left foot    :   max = ' + str(lfoot_max) + ', min = ' + str(lfoot_min) + ', distance = ' + str(lfoot_max - lfoot_min))
	print('Right foot   :   max = ' + str(rfoot_max) + ', min = ' + str(rfoot_min) + ', distance = ' + str(rfoot_max - rfoot_min))
#	print('One stop feet distance:   max = ' + str(feet_dis_max) + ', min = ' + str(feet_dis_min))

	print("done.")


if __name__ == "__main__":
	main()
