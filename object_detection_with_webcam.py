import numpy as np
import json
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import copy

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['CUDA_VISIBLE_DEVICES']='1'

sys.path.append('/home/simon/Documents/project/blue_point/models/research')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# # Model preparation 
# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# ## Loader

def load_model(model_name):
    model_dir = '/home/simon/Documents/project/blue_point/tf_detection_model_zoo/' + str(model_name)
    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/simon/Documents/project/blue_point/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# # Detection
# Load an object detection model:

import pathlib
model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'#'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

# Check the model's input signature, it expects a batch of 3-color images of type uint8: 

print(detection_model.inputs)

# And retuns several outputs:

detection_model.output_dtypes


detection_model.output_shapes

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
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


import cv2
import threading

#read webcam
#cap = cv2.VideoCapture(0)

#read ipcam
#cap = cv2.VideoCapture('http://192.168.1.2:8081')


# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
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

URL = "http://192.168.1.10:8081"







def run_inference(model, URL):

    

    # 使用無窮迴圈擷取影像，直到按下Esc鍵結束
    #while True:
    # 使用 getframe 取得最新的影像
     #   I = ipcam.getframe()
    
        #cv2.imshow('Image', I)
        #if cv2.waitKey(1000) == 27:
        #    cv2.destroyAllWindows()
        #    ipcam.stop()
        #    break

    last_image_np = None  
    first_time = True
    while True:
        
        
        if first_time:
            first_time = False
            #ret, image_np = cap.read()
            # 連接攝影機
            cap = ipcamCapture(URL)


            # 啟動子執行緒
            cap.start()

            # 暫停1秒，確保影像已經填充
            time.sleep(1)
            image_np_o = cap.getframe()
            image_np = copy.deepcopy(image_np_o)
            last_image_np = copy.deepcopy(image_np_o)
        else:
            image_np_o = cap.getframe()
            image_np = copy.deepcopy(image_np_o)
           

        if type(image_np) != np.ndarray:
            time.sleep(5)
            first_time = True
            cap.stop()
            print('Video capture restart!!!')
            continue

        if np.array_equal(image_np_o, last_image_np) and first_time == False : 
            continue
        else:
            last_image_np = copy.deepcopy(image_np_o)

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
        min_score_thresh=.6
        # iterate over all objects found


        find_object = [] 
        find = False
        system_time = time.time()
        show_time = str(time.ctime(system_time) + '_' + str(system_time)) 
        print( '-----' + show_time + '-----')
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn 
                class_name = category_index[output_dict['detection_classes'][i]]['name']
                print (class_name, boxes[i].tolist())
                # get position from here.


#                find_object.append([class_name,boxes[i].tolist()])
#                if class_name == 'person' and find == False:
#                    find = True
#
#        if find:
#            #print(find_object)
#            print('Saving image to save_images_tags/' + show_time + '.jpg/.json')
#            with open( 'save_images_tags/' + show_time + '.json', 'w', encoding='utf-8') as f:
#                json.dump({'find_objects':find_object}, f, ensure_ascii=False, indent=4)
#
#            #print(find_object)
#            #print(json.dumps({'find_objects':find_object}))
#            cv2.imwrite( 'save_images/' + show_time  + '.jpg', image_np_o)
#
#        #cv2.imshow('object_detection', cv2.resize(image_np, (1280, 960)))
#        #if cv2.waitKey(25) & 0xFF == ord('q'):
#        #    #cap.release()
#        #    cv2.destroyAllWindows()
#        #    cap.stop()
#        #    break

run_inference(detection_model, URL)
