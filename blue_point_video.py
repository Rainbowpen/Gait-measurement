## Object Detection for find blue point
#
#### depend tensorflow version == 2.x

from fun import *
import math
import tensorflow as tf
import cv2
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import find_peaks

captype = 'video' # camera, video or stream

def restart_cap():
		# 連接攝影機
		video_cap = ipcamCapture(config.URL)
		# 啟動子執行緒
		video_cap.start()

		# 暫停1秒，確保影像已經填充
		time.sleep(1)
		image_o = cap.getframe()
		image = copy.deepcopy(image_o)
		last_image = copy.deepcopy(image_o)
		return video_cap


def main():
	# main
	print('>>============================================================================<<')
	print("Getting data...")
	detection_model = run.load_model(config.MODEL_NAME)
	if captype == 'video':
		video_cap = cv2.VideoCapture(config.TEST_VIDEO_PATH)
		success, image = video_cap.read()
		if success:
			print('Find video.')
		else:
			print('Video not find.')
	elif captype == 'camera':
		video_cap = cv2.VideoCapture(config.CAMERA)
		success, image = video_cap.read()
		if success:
			print('Find camera.')
		else:
			print('Camera not found.')
	elif captype == 'stream':
		success = True
		last_image = None 
		video_cap = ipcamCapture(config.URL)
		vidoe_cap = restart_cap()

	image_array = []
	lfoot_data = []
	rfoot_data = []
	first_time = True
	while success:
		if captype == 'stream':
			image_o = video_cap.getframe()
			image = copy.deepcopy(image_o)
			
			if type(image) != np.ndarray:
				time.sleep(1)
				video_cap.stop()
				video_cap = restart_cap()
				print('Video capture restart!!!')
				continue

			if np.array_equal(image_o, last_image):
				continue
			else:
				last_image = copy.deepcopy(image_o)

		new_img, find_object = run.inference(detection_model, image)
		data = run.check_is_two_foot(find_object)
		if data is not None:
			lfoot_data.append(data[0])
			rfoot_data.append(data[1])
		
		height = new_img.shape[0]
		width = new_img.shape[1]
		#layers = new_img.shape[2]
		size = (width, height)
		image_array.append(new_img)
		
		if captype == 'video' or captype == 'camera':
			success, image = video_cap.read()

#		cv2.imshow('object_detection', cv2.resize(new_img, (640, 480)))
#		if cv2.waitKey(1) & 0xFF == ord('q'):
#			if captype == 'stream':
#				video_cap.release()
#				video_cap.stop()
#			cv2.destroyAllWindows()
#			break

	if captype == 'video' or captype == 'camera':
		#print(len(image_array))
		new_image_array = run.write_text_to_image(image_array)
		#print(type(new_image_array))
		#print(len(new_image_array))
		run.write_to_video(config.VIDEO_SAVE_PATH, new_image_array, size)

	if config.MODEL_NAME != 'blue_point':
		print('done.')
		return 0
	print("ok")
	
	print("Data calculating...")
	lfoot_data_xmax = []
	lfoot_data_ymax = []
	lfoot_data_xmin = []
	lfoot_data_ymin = []
	lfoot_data_top_x = []
	
	rfoot_data_xmax = []
	rfoot_data_ymax = []
	rfoot_data_xmin = []
	rfoot_data_ymin = []
	rfoot_data_top_x = []

	feet_distance = []
	for lf, rf in zip(lfoot_data, rfoot_data):
		lfoot_data_xmax.append(lf[2])
		lfoot_data_ymax.append(lf[3])
		lfoot_data_xmin.append(lf[0])
		lfoot_data_ymin.append(lf[1])
		lfoot_data_top_x.append((lf[2] - lf[0])/2 + lf[0])

		rfoot_data_xmax.append(rf[2])
		rfoot_data_ymax.append(rf[3])
		rfoot_data_xmin.append(rf[0])
		rfoot_data_ymin.append(rf[1])
		rfoot_data_top_x.append((rf[2] - rf[0])/2 + rf[0])
		feet_distance.append(abs(lf[3] - rf[3]))

	lfoot_max = max(lfoot_data_ymax)
	lfoot_min = min (lfoot_data_ymax)
	rfoot_max = max(rfoot_data_ymax)
	rfoot_min = min(rfoot_data_ymax)
	lfoot_avg = (lfoot_max - lfoot_min)/2 + lfoot_min
	rfoot_avg = (rfoot_max - rfoot_min)/2 + rfoot_min


	def convert_list_to_negative(data):
		new_data = []
		for i in data:
			new_data.append(0 - i)
		return new_data


	def get_step(x, y, index):
		foot_step_x = []
		foot_step_y = []
		for i in index:
			foot_step_x.append(x[i])
			foot_step_y.append(y[i])
		return foot_step_x, foot_step_y
		

	lfoot_step_y_index = find_peaks(lfoot_data_ymax, prominence=0.1)[0]
	rfoot_step_y_index = find_peaks(rfoot_data_ymax, prominence=0.1)[0]
	lfoot_step_x, lfoot_step_y = get_step(lfoot_data_top_x, lfoot_data_ymax, lfoot_step_y_index)
	rfoot_step_x, rfoot_step_y = get_step(rfoot_data_top_x, rfoot_data_ymax, rfoot_step_y_index)

	lfoot_data_ymax_negative = convert_list_to_negative(lfoot_data_ymax)
	rfoot_data_ymax_negative = convert_list_to_negative(rfoot_data_ymax)
	lfoot_step_y_index_negative =find_peaks(lfoot_data_ymax_negative, prominence=0.1)[0] 
	rfoot_step_y_index_negative =find_peaks(rfoot_data_ymax_negative, prominence=0.1)[0] 
	lfoot_step_x_min, lfoot_step_y_min = get_step(lfoot_data_top_x, lfoot_data_ymax, lfoot_step_y_index_negative)
	rfoot_step_x_min, rfoot_step_y_min = get_step(rfoot_data_top_x, rfoot_data_ymax, rfoot_step_y_index_negative)


	def step_up(max_index, min_index, foot_data_x, foot_data_y):
		foot_step_x_up = []
		foot_step_y_up = []
		
		for cnt, (i1, i2) in enumerate(zip(max_index, min_index)):
		#	if i1 - i2 < 1 and cnt != 0:
		#		foot_step_x_up.append(foot_data_x[min_index[cnt - 1]:i1+1])
		#		foot_step_y_up.append(foot_data_y[min_index[cnt - 1]:i1+1])
		#	elif i1 - i2 < 1:
		#		continue
		#	else:
		#		foot_step_x_up.append(foot_data_x[i2:i1+1])
		#		foot_step_y_up.append(foot_data_y[i2:i1+1])
			foot_step_x_up.append([foot_data_x[i1]])
			foot_step_y_up.append([foot_data_y[i1]])
		return foot_step_x_up, foot_step_y_up

	lfoot_step_x_up, lfoot_step_y_up = step_up(lfoot_step_y_index, lfoot_step_y_index_negative, lfoot_data_top_x, lfoot_data_ymax)
	rfoot_step_x_up, rfoot_step_y_up = step_up(rfoot_step_y_index, rfoot_step_y_index_negative, rfoot_data_top_x, rfoot_data_ymax)


	def get_track(foot_step_x_up, foot_step_y_up, move):
	# must fix this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# return x, y track and step dat
		x_track = []
		y_track = []
		x_step_dat = []
		y_step_dat = []
		y_add = move
		#y_add = 0
		speed = 0.08
		for cnt, (x_up, y_up) in enumerate(zip(foot_step_x_up, foot_step_y_up)):
			for cnt2, (x, y) in enumerate(zip(x_up, y_up)):
				x_track.append(x)
				y_track.append((y - y_up[0]) + y_add)
				if cnt2 + 1 == len(x_up):
					x_step_dat.append(x)
					y_step_dat.append((y - y_up[0]) + y_add)
					#y_add += y_up[-1]
					y_add += speed

		return [x_track, y_track], [x_step_dat, y_step_dat]

	lmove = (lfoot_data_ymax[lfoot_step_y_index[0]] - rfoot_data_ymax[rfoot_step_y_index[0]])# * 2
	print("debug: lmove is " + str(lmove))

	lf_track, lf_step_dat = get_track(lfoot_step_x_up, lfoot_step_y_up, lmove)
	rf_track, rf_step_dat = get_track(rfoot_step_x_up, rfoot_step_y_up, 0)

	#print(lf_track)
	#print(rf_track)


	lf_distance_cm = []
	rf_distance_cm = []
	
	for cnt, (lf, rf, lfm, rfm) in enumerate(zip(lfoot_step_y, rfoot_step_y, lfoot_step_y_min, rfoot_step_y_min)):
		if cnt == 0:
			continue
		lf_distance_cm.append(int(((abs(lf - lfm) / config.RULER) * config.CM)))
		rf_distance_cm.append(int(((abs(rf - rfm) / config.RULER) * config.CM)))
	print('left  : ' + str(lf_distance_cm))
	print('right : ' + str(rf_distance_cm))



	#print('Left foot    :   max = ' + str(lfoot_max) + ', min = ' + str(lfoot_min) + ', distance = ' + str(lfoot_max - lfoot_min))
	#print('Right foot   :   max = ' + str(rfoot_max) + ', min = ' + str(rfoot_min) + ', distance = ' + str(rfoot_max - rfoot_min))
	#print('One stop feet distance:   max = ' + str(feet_dis_max) + ', min = ' + str(feet_dis_min))
	draw_ymax = max(lf_track[1][-1], rf_track[1][-1])# - 0.3
	draw_xmax = max(lf_track[0][-1], rf_track[0][-1])

	def upside_down(var):
		new_var = []
		for i in var:
			#new_var = 
			pass


	def x_track_avg(x, size):
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


	y_ticks = []
	y_cnt = 0
	while y_cnt <= draw_ymax:
		y_ticks.append(str(((y_cnt//0.45)*40)/100) + ' M')
		y_cnt = y_cnt + 1


	lf_cm_y = []
	for i in range(1, len(lfoot_data_ymax) + 1):
		lf_cm_y.append(i)
	rf_cm_y = []
	for i in range(1, len(rfoot_data_ymax) + 1):
		rf_cm_y.append(i)
	print("ok")

	print("Drawing data to images...")
	avg_size = 1
	lf_x_track_avg = x_track_avg(lf_track[0], avg_size)
	rf_x_track_avg = x_track_avg(rf_track[0], avg_size)
	plt.style.use('bmh')
	fig = plt.figure(figsize=(5,20))
	plt.axes()
	plt.xlim(1, 0)
	#plt.ylim(0, draw_ymax)
	#plt.xticks(range(len(x_ticks)), x_ticks)#, minor=True, rotation=0)
	plt.xticks(range(2), [(str(((1//0.45)*40)/100)) + ' M', '0 M'])#, minor=True, rotation=0)
	plt.yticks(range(len(y_ticks)), y_ticks)#, minor=True)
	#plt.yticks(range(1), ['0,0'], rotation=0)
	plt.title('Feet track')
	plt.plot(lf_x_track_avg[:], lf_track[1][avg_size-1:], ':g')#, label='Left foot')
	plt.plot(rf_x_track_avg[:], rf_track[1][avg_size-1:], ':r')#, label='Right foot')
	plt.plot(lf_step_dat[0], lf_step_dat[1], 'og', label='Left foot')
	plt.plot(rf_step_dat[0], rf_step_dat[1], 'or', label='Reft foot')
	plt.legend(loc = 'lower left')
	#plt.grid(color='r', linestyle='-', linewidth=0.2)
	#plt.grid(which='minor', alpha=0.2)
	#plt.tight_layout()
	plt.savefig('./track.png', dpi=400)

	# track 2
	fig = plt.figure(figsize=(5,20))
	plt.axes()
	plt.xlim(1, 0)
	#plt.ylim(0, draw_ymax)
	#plt.xticks(range(len(x_ticks)), x_ticks)#, minor=True, rotation=0)
	plt.xticks(range(2), [(str(((1//0.45)*40)/100)) + ' M', '0 M'])#, minor=True, rotation=0)
	plt.yticks(range(len(y_ticks)), y_ticks)#, minor=True)
	#plt.yticks(range(1), ['0,0'], rotation=0)
	plt.title('Feet track')
	plt.plot(lf_x_track_avg[:], lf_track[1][avg_size-1:], ':g')#, label='Left foot')
	plt.plot(rf_x_track_avg[:], rf_track[1][avg_size-1:], ':r')#, label='Right foot')
	plt.plot(lf_step_dat[0], lf_step_dat[1], 'og', label='Left foot')
	plt.plot(rf_step_dat[0], rf_step_dat[1], 'or', label='Reft foot')
	plt.legend(loc = 'lower left')
	plt.savefig('./track2.png', dpi=400)




	fig4 = plt.figure(figsize=(20,5))
	ax2 = fig4.add_subplot(1, 1, 1)
	ax2.set_title('Feet step size scatter plot')
	#plt.xticks(range(1), '')
	#plt.yticks(range(1), '')
	plt.plot(lf_cm_y, lfoot_data_ymax, 'g', label='Left foot')
	plt.plot(rf_cm_y, rfoot_data_ymax, 'r', label='Right foot')
	ax2.set_ylabel('Y')
	ax2.set_xlabel('Time')
	ax2.legend(['Left foot', 'Right foot'], loc='upper left')
	plt.savefig('./step_size_scatter_plot.png')
	print("ok")

	print("Creating animation...")
	# feet moving gif
	#fig_gif, ax_gif= plt.subplots()#figsize=(20,20))
	#plt.subplots_adjust(left=0, right=0.1, top=0.5, bottom=0)
	fig_gif = plt.figure(figsize=(8, 10), dpi=100)
	#fig_gif.set_tight_layout(True)
	#plt.title('Data visualization')
	ax_gif = fig_gif.add_subplot(8, 1, (1, 5))
	lf_gif, = plt.plot([], [], 'go', label='Left foot')
	rf_gif, = plt.plot([], [], 'ro', label='Reft foot')

	ax_gif2 = fig_gif.add_subplot(8, 1, (7, 8))
	lf_gif2 = plt.plot(lf_cm_y, lfoot_data_ymax, 'g', label='Left foot')
	rf_gif2 = plt.plot(rf_cm_y, rfoot_data_ymax, 'r', label='Right foot')
	time_gif2, = plt.plot([], [], 'b-', label='Time')
	

	def gif_init():  
		ax_gif.set_xlim(0, 1)  
		ax_gif.set_ylim(0, 1)  
		ax_gif.set_ylabel('Y')
		ax_gif.set_xlabel('X')
		ax_gif.legend(['Left foot', 'Right foot'], loc=4)
		
		ax_gif2.set_ylabel('Y')
		ax_gif2.set_xlabel('Number of data')
		ax_gif2.legend(['Left foot','Right foot'], loc=4)


	def gif_update(i):
		fl = i
		lf_gif.set_data(1 - lfoot_data_top_x[fl], lfoot_data_ymax[fl])  
		rf_gif.set_data(1 - rfoot_data_top_x[fl], rfoot_data_ymax[fl])  
		time_gif2.set_data([fl, fl], [0, 1])  

		return lf_gif, rf_gif, time_gif2

	ani = FuncAnimation(fig_gif, gif_update, frames=np.arange(0, len(lfoot_data_top_x)), init_func=gif_init)#, blit=True)

	# save animation at 7 frames per second 
	ani.save("feet_tracking.gif", writer='imagemagick', fps=7)  
	print("ok")

	print("done.")


if __name__ == "__main__":
	main()
