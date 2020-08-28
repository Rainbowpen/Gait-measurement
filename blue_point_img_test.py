## Object Detection for find blue point
#
#### depend tensorflow version == 2.x


import numpy as np
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
#print(sys.path)
#print('---------')
#print(sys.executable)

# for tf_1.0
# 将程序限定在一块GPU上
#from tensorflow.python.keras import backend as K
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# TF 1.x
#config = tf.ConfigProto(intra_op_parallelism_threads=1,
#                         inter_op_parallelism_threads=1)
#config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
#K.set_session(tf.Session(config=config))


# for tf 2.x
# Sovle GPU problem
#gpus= tf.config.list_physical_devices('GPU') 
#print(gpus) 
#tf.config.experimental.set_memory_growth(gpus[0], True) 
#os.environ['CUDA_VISIBLE_DEVICES']='1' 


# Import the object detection module.
#sys.path.append('/home/simon/Documents/project/blue_point/models/research')
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
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./my/images/test')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)


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

def show_inference(model, image_path):
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	image_np = np.array(Image.open(image_path))
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
	# iterate over all objects found
	for i in range(min(max_boxes_to_draw, boxes.shape[0])):
		if scores is None or scores[i] > min_score_thresh:
			# boxes[i] is the box which will be drawn
			class_name = category_index[output_dict['detection_classes'][i]]['name']
			print (class_name, boxes[i])

	return image_np


for image_path in TEST_IMAGE_PATHS:
	print('and here')
	print(str(image_path) + ' :')
	new_img = show_inference(detection_model, image_path)
	#new_img = cv2.resize(image_np, (960, 540))
	new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
	cv2.imshow('Color image', new_img)
	cv2.waitKey(0)
	print('')




