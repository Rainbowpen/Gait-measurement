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
from scipy.signal import find_peaks
#from memory_profiler import profile

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


	#@profile # for memory monitor
	def write_text_to_image(image_array, walking_pace, step_width, left, right):
		# image_array is array of image and this funtion will
		# write text on it.

		from sys import getsizeof
		import gc
		from PIL import Image, ImageDraw, ImageFont

		for i in range(len(image_array)):
			new_image = np.array(image_array[i], dtype=np.uint8)
			new_image = Image.fromarray(new_image)
			draw = ImageDraw.Draw(new_image)
			#font = ImageFont.truetype(size=20)
			draw.text((10,10), "Frame number: " + str(i), fill=(0, 0, 0))
			draw.text((10,25), "Walking pace: " + str(walking_pace[i]) + " (m/s)", fill=(0, 0, 0))
			draw.text((10,40), "Step width: " + str(step_width[i]) + " (cm)", fill=(0, 0, 0))
			draw.text((10,55), "Stride Length: ", fill=(0, 0, 0))
			draw.text((25,70), "Left: " + str(left[i]) + " (cm)", fill=(0, 0, 0))
			draw.text((25,85), "Right: " + str(right[i]) + " (cm)", fill=(0, 0, 0))

			new_image_array = np.frombuffer(new_image.tobytes(), dtype=np.uint8)
			new_image = new_image_array.reshape((new_image.size[1], new_image.size[0], 3))
			image_array[i] = new_image 

			#del new_image, draw, new_image_array
			#if i%10 == 0:
			#	gc.collect()
			#print(i)

		return image_array


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


class Feet:
	# feet's thing

	def __init__(self, lfoot_data, rfoot_data):
		self.lfoot_data = lfoot_data
		self.rfoot_data = rfoot_data
		self.lfoot_data_xmax = []
		self.lfoot_data_ymax = []
		self.lfoot_data_xmin = []
		self.lfoot_data_ymin = []
		self.lfoot_data_top_x = []
	
		self.rfoot_data_xmax = []
		self.rfoot_data_ymax = []
		self.rfoot_data_xmin = []
		self.rfoot_data_ymin = []
		self.rfoot_data_top_x = []

		self.feet_distance = []

		self.update()


	def get_data_to_var(self):

		for lf, rf in zip(self.lfoot_data, self.rfoot_data):
			self.lfoot_data_xmax.append(lf[2])
			self.lfoot_data_ymax.append(lf[3])
			self.lfoot_data_xmin.append(lf[0])
			self.lfoot_data_ymin.append(lf[1])
			
			self.rfoot_data_xmax.append(rf[2])
			self.rfoot_data_ymax.append(rf[3])
			self.rfoot_data_xmin.append(rf[0])
			self.rfoot_data_ymin.append(rf[1])
			if lf[2] is not None:
				self.lfoot_data_top_x.append((lf[2] - lf[0])/2 + lf[0])
				self.rfoot_data_top_x.append((rf[2] - rf[0])/2 + rf[0])
				self.feet_distance.append(abs(lf[3] - rf[3]))
			else:
				self.lfoot_data_top_x.append(None)
				self.rfoot_data_top_x.append(None)
				self.feet_distance.append(None)
	
	
		lfoot_data_ymax_np = np.array(self.lfoot_data_ymax, dtype=np.float64)	
		self.lfoot_max = np.nanmax(lfoot_data_ymax_np)
		self.lfoot_min = np.nanmin(lfoot_data_ymax_np)
		



	def update(self):
		# Feet class main

		self.get_data_to_var()
		rfoot_data_ymax_np = np.array(self.rfoot_data_ymax, dtype=np.float64)	
		self.rfoot_max = np.nanmax(rfoot_data_ymax_np)
		self.rfoot_min = np.nanmin(rfoot_data_ymax_np)
		
		self.lfoot_avg = (self.lfoot_max - self.lfoot_min)/2 + self.lfoot_min
		self.rfoot_avg = (self.rfoot_max - self.rfoot_min)/2 + self.rfoot_min

		self.convert_and_get_step()
		self.lfoot_step_x_up, self.lfoot_step_y_up = self.step_up(self.lfoot_step_y_index,
													self.lfoot_step_y_index_negative, 
													self.lfoot_data_top_x, 
													self.lfoot_data_ymax)
		self.rfoot_step_x_up, self.rfoot_step_y_up = self.step_up(self.rfoot_step_y_index,
													self.rfoot_step_y_index_negative, 
													self.rfoot_data_top_x, 
													self.rfoot_data_ymax)
		
		#print(self.get_list_without_none(self.lfoot_data_ymax)[self.lfoot_step_y_index[0]])
		#print(self.lfoot_step_y_index)
		self.lmove = (self.get_list_without_none(self.lfoot_data_ymax)[self.lfoot_step_y_index[0]] - self.get_list_without_none(self.rfoot_data_ymax)[self.rfoot_step_y_index[0]])
		#print("debug: lmove is " + str(self.lmove))

		self.lf_track, self.lf_step_dat = self.get_track(self.lfoot_step_x_up, 
												self.lfoot_step_y_up, self.lmove)
		self.rf_track, self.rf_step_dat = self.get_track(self.rfoot_step_x_up, 
												self.rfoot_step_y_up, 0)


		self.draw_ymax = max(self.get_list_without_none(self.lf_track[1])[-1], self.get_list_without_none(self.rf_track[1])[-1])
		self.draw_xmax = max(self.get_list_without_none(self.lf_track[0])[-1], self.get_list_without_none(self.rf_track[0])[-1])




	def get_list_without_none(self, none_list):
		# return list but without None.
		return [i for i in none_list if i] 


	def get_list_index_without_none(self, none_list):
		# return list index but without None index.
		not_none_index = []
		for cnt, i in enumerate(none_list):
			if i is not None:
				not_none_index.append(cnt)

		#print("not_none_index: " + str(not_none_index))
		return not_none_index


	def convert_list_to_negative(self, data):
		new_data = []
		for i in data:
			if i is not None:
				new_data.append(0 - i)
			else:
				new_data.append(None)
		return new_data


	def get_step(self, x, y, index):
		foot_step_x = []
		foot_step_y = []
		for i in index:
			foot_step_x.append(x[i])
			foot_step_y.append(y[i])

		return foot_step_x, foot_step_y


	def convert_no_none_list_index_to_real(self, no_none_list_index, source_list):
		
		real_list_index = []
		rl = self.get_list_index_without_none(source_list)
		for i in no_none_list_index:
			real_list_index.append(rl[i - 1])

		return real_list_index



	def convert_and_get_step(self):
		lfoot_step_y_index_no_none = find_peaks(self.get_list_without_none(self.lfoot_data_ymax), prominence=0.1)[0]
		self.lfoot_step_y_index = self.convert_no_none_list_index_to_real(lfoot_step_y_index_no_none, self.lfoot_data_ymax)

		rfoot_step_y_index_no_none = find_peaks(self.get_list_without_none(self.rfoot_data_ymax), prominence=0.1)[0]
		self.rfoot_step_y_index = self.convert_no_none_list_index_to_real(rfoot_step_y_index_no_none, self.rfoot_data_ymax)

		self.lfoot_step_x, self.lfoot_step_y = self.get_step(self.lfoot_data_top_x, self.lfoot_data_ymax, self.lfoot_step_y_index)
		self.rfoot_step_x, self.rfoot_step_y = self.get_step(self.rfoot_data_top_x, self.rfoot_data_ymax, self.rfoot_step_y_index)
	
		self.lfoot_data_ymax_negative = self.convert_list_to_negative(self.lfoot_data_ymax)
		self.rfoot_data_ymax_negative = self.convert_list_to_negative(self.rfoot_data_ymax)
		
		lfoot_step_y_index_negative_no_none = find_peaks(self.get_list_without_none(self.lfoot_data_ymax_negative), prominence=0.1)[0] 
		self.lfoot_step_y_index_negative = self.convert_no_none_list_index_to_real(lfoot_step_y_index_no_none, self.lfoot_data_ymax_negative)
		
		rfoot_step_y_index_negative_no_none = find_peaks(self.get_list_without_none(self.rfoot_data_ymax_negative), prominence=0.1)[0] 
		self.rfoot_step_y_index_negative = self.convert_no_none_list_index_to_real(rfoot_step_y_index_no_none, self.rfoot_data_ymax_negative)
		
		self.lfoot_step_x_min, self.lfoot_step_y_min = self.get_step(self.lfoot_data_top_x, self.lfoot_data_ymax, self.lfoot_step_y_index_negative)
		self.rfoot_step_x_min, self.rfoot_step_y_min = self.get_step(self.rfoot_data_top_x, self.rfoot_data_ymax, self.rfoot_step_y_index_negative)

	
	def step_up(self, max_index, min_index, foot_data_x, foot_data_y):
		# I forget what it is.

		foot_step_x_up = []
		foot_step_y_up = []
		for cnt, (i1, i2) in enumerate(zip(max_index, min_index)):
			foot_step_x_up.append([foot_data_x[i1]])
			foot_step_y_up.append([foot_data_y[i1]])
		return foot_step_x_up, foot_step_y_up



	def get_track(self, foot_step_x_up, foot_step_y_up, move):
	# must fix this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# return x, y track and step dat
		x_track = []
		y_track = []
		x_step_dat = []
		y_step_dat = []
		y_add = move
		#y_add = 0
		speed = 0.08
		#print(foot_step_x_up)
		#print(foot_step_y_up)
		for cnt, (x_up, y_up) in enumerate(zip(foot_step_x_up, foot_step_y_up)):
			for cnt2, (x, y) in enumerate(zip(x_up, y_up)):
				if x is not None:	
					x_track.append(x)
					y_track.append((y - y_up[0]) + y_add)
					if cnt2 + 1 == len(x_up):
						x_step_dat.append(x)
						y_step_dat.append((y - y_up[0]) + y_add)
						#y_add += y_up[-1]
						y_add += speed
				else:
					x_track.append(None)
					y_track.append(None)
					x_step_dat.append(None)
					y_step_dat.append(None)

		return [x_track, y_track], [x_step_dat, y_step_dat]


	def x_track_avg(self, x, size):
		x_track = []
		if size <= 1:
			return x

		for i in range(size - 1, len(x)):
			avg = 0 
			for j in range(i - size, i):
				avg += x[j]
			avg = avg / size
			x_track.append(avg)

		return x_track


