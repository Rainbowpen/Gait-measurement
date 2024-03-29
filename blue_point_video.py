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
		else:
			lfoot_data.append([None, None, None, None])
			rfoot_data.append([None, None, None, None])
	
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


	if config.MODEL_NAME != 'blue_point':
		print('done.')
		return 0
	print("ok")
	
	
	print("Data calculating...")
	feet = Feet(lfoot_data, rfoot_data)

		

	





#	lf_distance_cm = []
#	rf_distance_cm = []
#	
#	for cnt, (lf, rf, lfm, rfm) in enumerate(zip(lfoot_step_y, rfoot_step_y, lfoot_step_y_min, rfoot_step_y_min)):
#		if cnt == 0:
#			continue
#		lf_distance_cm.append(int(((abs(lf - lfm) / config.RULER) * config.CM)))
#		rf_distance_cm.append(int(((abs(rf - rfm) / config.RULER) * config.CM)))
#	print('left  : ' + str(lf_distance_cm))
#	print('right : ' + str(rf_distance_cm))



	#print('Left foot    :   max = ' + str(lfoot_max) + ', min = ' + str(lfoot_min) + ', distance = ' + str(lfoot_max - lfoot_min))
	#print('Right foot   :   max = ' + str(rfoot_max) + ', min = ' + str(rfoot_min) + ', distance = ' + str(rfoot_max - rfoot_min))
	#print('One stop feet distance:   max = ' + str(feet_dis_max) + ', min = ' + str(feet_dis_min))
	
#	def upside_down(var):
#		new_var = []
#		for i in var:
#			#new_var = 
#			pass


	

	y_ticks = []
	y_cnt = 0
	while y_cnt <= feet.draw_ymax:
		y_ticks.append(str(((y_cnt//0.45)*40)/100) + ' M')
		y_cnt = y_cnt + 1


	#lf_cm_y = []
	#for i in range(1, len(feet.lfoot_data_ymax) + 1):
	#	lf_cm_y.append(i)
	#rf_cm_y = []
	#for i in range(1, len(feet.rfoot_data_ymax) + 1):
	#	rf_cm_y.append(i)
	print("ok")

	print("Drawing data to images...")
	avg_size = 1
	lf_x_track_avg = feet.x_track_avg(feet.lf_track[0], avg_size)
	rf_x_track_avg = feet.x_track_avg(feet.rf_track[0], avg_size)
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
	plt.plot(lf_x_track_avg[:], feet.lf_track[1][avg_size-1:], ':g')#, label='Left foot')
	plt.plot(rf_x_track_avg[:], feet.rf_track[1][avg_size-1:], ':r')#, label='Right foot')
	plt.plot(feet.lf_step_dat[0], feet.lf_step_dat[1], 'og', label='Left foot')
	plt.plot(feet.rf_step_dat[0], feet.rf_step_dat[1], 'or', label='Reft foot')
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

	plt.plot(lf_x_track_avg[:], feet.lf_track[1][avg_size-1:], ':g')#, label='Left foot')
	plt.plot(rf_x_track_avg[:], feet.rf_track[1][avg_size-1:], ':r')#, label='Right foot')
	plt.plot(feet.lf_step_dat[0], feet.lf_step_dat[1], 'og', label='Left foot')
	plt.plot(feet.rf_step_dat[0], feet.rf_step_dat[1], 'or', label='Reft foot')

	plt.legend(loc = 'lower left')
	plt.savefig('./track2.png', dpi=400)



	fig4 = plt.figure(figsize=(20,5))
	ax2 = fig4.add_subplot(1, 1, 1)
	ax2.set_title('Feet step size scatter plot')
	#plt.xticks(range(1), '')
	#plt.yticks(range(1), '')
	print(len(feet.get_list_index_without_none(feet.lfoot_data_ymax)))
	plt.plot(feet.get_list_index_without_none(feet.lfoot_data_ymax),
							feet.get_list_without_none(feet.lfoot_data_ymax), 
							'g', label='Left foot')
	plt.plot(feet.get_list_index_without_none(feet.rfoot_data_ymax), 
							feet.get_list_without_none(feet.rfoot_data_ymax),
							'r', label='Right foot')
	ax2.set_ylabel('Y')
	ax2.set_xlabel('Time')
	ax2.legend(['Left foot', 'Right foot'], loc='upper left')
	plt.savefig('./step_size_scatter_plot.png')
	print("ok")

	print("Creating animation...")
	# feet moving gif
	#fig_gif, ax_gif= plt.subplots()#figsize=(20,20))
	#plt.subplots_adjust(left=0, right=0.1, top=0.5, bottom=0)
	fig_gif = plt.figure(figsize=(8, 10), dpi=20)

	#fig_gif.set_tight_layout(True)
	#plt.title('Data visualization')
	ax_gif = fig_gif.add_subplot(8, 1, (1, 5))
	lf_gif, = plt.plot([], [], 'go', label='Left foot')
	rf_gif, = plt.plot([], [], 'ro', label='Reft foot')

	ax_gif2 = fig_gif.add_subplot(8, 1, (7, 8))
	lf_gif2 = plt.plot(feet.get_list_index_without_none(feet.lfoot_data_ymax),
						feet.get_list_without_none(feet.lfoot_data_ymax),
						'g', label='Left foot')
	rf_gif2 = plt.plot(feet.get_list_index_without_none(feet.rfoot_data_ymax),
						feet.get_list_without_none(feet.rfoot_data_ymax),
						'r', label='Right foot')
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
		if feet.lfoot_data_top_x[fl] is not None:
			lf_gif.set_data(1 - feet.lfoot_data_top_x[fl], feet.lfoot_data_ymax[fl])  
			rf_gif.set_data(1 - feet.rfoot_data_top_x[fl], feet.rfoot_data_ymax[fl])  

		time_gif2.set_data([fl, fl], [0, 1])  

		return lf_gif, rf_gif, time_gif2

	#ani = FuncAnimation(fig_gif, gif_update, frames=np.arange(0, len(feet.lfoot_data_top_x)), init_func=gif_init)#, blit=True)

	# save animation at 7 frames per second 
	#ani.save("feet_tracking.gif", writer='imagemagick', fps=7)  
	print("ok")

	print("Saving video...")
	if captype == 'video' or captype == 'camera':
		#image_array = run.write_text_to_image(image_array, walking_pace, 
		#											step_width, left, right)
		run.write_to_video(config.VIDEO_SAVE_PATH, image_array, size)

	print("done.")


if __name__ == "__main__":
	main()
