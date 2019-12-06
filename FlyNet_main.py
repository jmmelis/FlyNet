from __future__ import print_function
import sys
import vtk
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import os
import os.path
import math
import copy
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from session_select import Ui_Session_Dialog

from fly_net_ui import Ui_MainWindow

class CheckableDirModel(QtGui.QDirModel):
	def __init__(self, parent=None):
	    QtGui.QDirModel.__init__(self, None)
	    self.checks = {}

	def data(self, index, role=QtCore.Qt.DisplayRole):
	    if role != QtCore.Qt.CheckStateRole:
	        return QtGui.QDirModel.data(self, index, role)
	    else:
	        if index.column() == 0:
	            return self.checkState(index)

	def flags(self, index):
	    return QtGui.QDirModel.flags(self, index) | QtCore.Qt.ItemIsUserCheckable

	def checkState(self, index):
	    if index in self.checks:
	        return self.checks[index]
	    else:
	        return QtCore.Qt.Unchecked

	def setData(self, index, value, role):
	    if (role == QtCore.Qt.CheckStateRole and index.column() == 0):
	        self.checks[index] = value
	        self.emit(QtCore.SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
	        return True 

	    return QtGui.QDirModel.setData(self, index, value, role)

# QDialog clasess:

class SelectFolderWindow(QtGui.QDialog, Ui_Session_Dialog):

	def __init__(self, directory, parent=None):
		super(SelectFolderWindow,self).__init__(parent)
		self.setupUi(self)
		self.folder_name = None
		self.folder_path = None
		self.file_model = QFileSystemModel()
		self.directory = directory
		self.file_model.setRootPath(directory)
		self.folder_tree.setModel(self.file_model)
		self.folder_tree.setRootIndex(self.file_model.index(self.directory));
		self.folder_tree.clicked.connect(self.set_session_folder)

	def update_file_model(self,new_dir):
		self.directory = new_dir
		self.file_model.setRootPath(new_dir)

	def set_session_folder(self, index):
		indexItem = self.file_model.index(index.row(), 0, index.parent())
		self.folder_name = self.file_model.fileName(indexItem)
		self.folder_path = self.file_model.filePath(indexItem)
		self.selected_session.setText(self.folder_path)

class FlyNetViewer(QtWidgets.QMainWindow, Ui_MainWindow, QObject):

	def __init__(self, parent=None):
		super(FlyNetViewer,self).__init__(parent)
		self.setupUi(self)

		self.ds = 0.040
		self.cam_mag = 0.5
		self.frame_size = [[256,256],[256,256],[256,256]]

		self.scale_set = False
		self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.scale_L = 1.0
		self.scale_R = 1.0

		self.annotation_active = False

		self.load_data_gui()
		self.inactive_frame_gui()

		self.tabWidget.setTabEnabled(1, False)

	#-----------------------------------------------------------------------
	#
	#	Data loader:
	#
	#-----------------------------------------------------------------------

	def load_data_gui(self):
		self.base_dir = '/media/flyami/flyami_hdd_1'
		#self.select_session_window = SelectFolderWindow('/home')
		self.select_session_window = SelectFolderWindow(self.base_dir)
		self.select_session_window.setWindowTitle("Select session folder")
		#self.select_bckg_window = SelectFolderWindow('/home')
		self.select_bckg_window = SelectFolderWindow(self.base_dir)
		self.select_bckg_window.setWindowTitle("Select background folder")
		#self.select_calib_window = SelectFolderWindow('/home')
		self.select_calib_window = SelectFolderWindow(self.base_dir)
		self.select_calib_window.setWindowTitle("Select calibration file")
		#self.select_movie_window = SelectFolderWindow('/home')
		self.select_movie_window = SelectFolderWindow(self.base_dir)
		self.select_movie_window.setWindowTitle("Select movie folder")
		#self.select_weights_window = SelectFolderWindow('/home')
		self.select_weights_window = SelectFolderWindow('/home')
		self.select_weights_window.setWindowTitle("Select network weights file")
		# Parameters
		self.session_path = None
		self.session_folder = None
		self.bckg_path = None
		self.bckg_folder = None
		self.bckg_frames = []
		self.bckg_img_format = None
		self.calib_path = None
		self.calib_file = None
		self.N_cam = None
		self.camera_folders = []
		self.frame_name = None
		self.frame_img_format = None
		self.N_mov = None
		self.mov_folder_list = []
		self.trigger_mode = None
		self.start_frame = None
		self.trigger_frame = None
		self.end_frame = None
		self.N_frames_list = []
		# Additional parameters
		self.trig_modes = ['...','start','center','end']
		self.network_options = ['...','network_27_11']
		# Start GUI
		self.select_seq_btn.clicked.connect(self.select_session_callback)
		self.seq_folder_disp.setText('...')
		self.select_bckg_btn.clicked.connect(self.select_bckg_callback)
		self.bckg_folder_disp.setText('...')
		self.select_calib_btn.clicked.connect(self.select_calib_callback)
		self.calib_folder_disp.setText('...')
		self.add_movie_btn.clicked.connect(self.select_mov_callback)
		self.reset_movie_btn.clicked.connect(self.reset_mov_list_callback)
		self.load_seq_btn.clicked.connect(self.load_seq_callback)
		self.set_table_widget(self.movie_table)
		self.set_trigger_mode_box(self.trigger_mode_combo)

	def select_session_callback(self):
		self.select_session_window.exec_()
		self.session_path = str(self.select_session_window.folder_path)
		print(self.session_path)
		self.session_folder = str(self.select_session_window.folder_name)
		self.seq_folder_disp.setText(self.session_folder)
		self.select_bckg_window.update_file_model(self.session_path)
		self.select_calib_window.update_file_model(self.session_path)
		self.select_movie_window.update_file_model(self.session_path)

	def select_bckg_callback(self):
		self.select_bckg_window.exec_()
		self.bckg_path = self.select_bckg_window.folder_path
		self.bckg_folder = self.select_bckg_window.folder_name
		for root, dirs, files in os.walk(self.bckg_path):
			if len(files)>0:
				for file in files:
					file_name = os.path.splitext(file)[0]
					file_ext = os.path.splitext(file)[1]
					if file_ext == '.tif':
						self.bckg_frames.append(file_name)
						self.bckg_img_format = '.tif'
					elif file_ext == '.bmp':
						self.bckg_frames.append(file_name)
						self.bckg_img_format = '.bmp'
					else:
						print('error: unknown image format')
				self.bckg_frames.sort()
		self.bckg_folder_disp.setText(self.bckg_folder)

	def select_calib_callback(self):
		self.select_calib_window.exec_()
		self.calib_path = self.select_calib_window.folder_path
		for root, dirs, files in os.walk(self.calib_path):
			if len(files)>0:
				for file in files:
					file_name = os.path.splitext(file)[0]
					file_ext = os.path.splitext(file)[1]
					if file_ext == '.txt':
						self.calib_file = file_name
		if self.calib_file == 'cam_calib':
			self.calib_folder_disp.setText(self.calib_file)
			self.LoadCalibrationMatrix()
			self.CalculateProjectionMatrixes(self.ds,self.cam_mag)
			self.img_viewer_1.load_camera_calibration(self.c_params,self.c2w_matrices,self.w2c_matrices)
		else:
			print('error: could not find cam_calib.txt')
			self.calib_folder_disp.setText('error!')

	def select_mov_callback(self):
		self.select_movie_window.exec_()
		mov_path = self.select_movie_window.folder_path
		mov_folder = self.select_movie_window.folder_name
		if not self.mov_folder_list:
			# Check camera folders:
			for root, dirs, files in os.walk(mov_path):
				if len(dirs)>0:
					for cam_folder in dirs:
						self.camera_folders.append(cam_folder)
			self.camera_folders.sort()
			self.N_cam = len(self.camera_folders)
			# Check frame format:
			frame_list = []
			for root, dirs, files in os.walk(mov_path + '/' + self.camera_folders[0]):
				if len(files)>0:
					for file in files:
						file_name = str(os.path.splitext(file)[0])
						file_ext = str(os.path.splitext(file)[1])
						if file_ext == '.tif':
							frame_nr = int(''.join(filter(str.isdigit, file_name)))
							if frame_nr >= 0:
								frame_list.append(frame_nr)
							if len(frame_list)==1:
								self.frame_name = file_name.replace(str(frame_nr),'')
								self.frame_img_format = '.tif'
						elif file_ext == '.bmp':
							frame_nr = int(''.join(filter(str.isdigit, file_name)))
							if frame_nr >= 0:
								frame_list.append(frame_nr)
							if len(frame_list)==1:
								self.frame_name = file_name.replace(str(frame_nr),'')
								self.frame_img_format = '.bmp'
						else:
							print('error: unknown image format')
			frame_list.sort()
			self.start_frame = frame_list[0]
			self.end_frame = frame_list[-1]
			self.calc_trigger_wb()
			self.mov_folder_list.append(mov_folder)
			self.N_mov = len(self.mov_folder_list)
		else:
			cam_folder_list = []
			for root, dirs, files in os.walk(mov_path):
				if len(dirs)>0:
					for cam_folder in dirs:
						cam_folder_list.append(cam_folder)
			if not set(self.camera_folders) == set(cam_folder_list):
				print('error: different set of camera folders')
			else:
				self.mov_folder_list.append(mov_folder)
				self.N_mov = len(self.mov_folder_list)
		self.update_table_widget()

	def reset_mov_list_callback(self):
		self.mov_folder_list = []
		self.cam_folder_list = []
		self.N_mov = 0
		self.N_cam = 0
		self.update_table_widget()

	def set_trigger_mode_box(self,combo_box_in):
		self.trigger_mode_box = combo_box_in
		self.trigger_mode_box.addItem(self.trig_modes[0])
		self.trigger_mode_box.addItem(self.trig_modes[1])
		self.trigger_mode_box.addItem(self.trig_modes[2])
		self.trigger_mode_box.addItem(self.trig_modes[3])
		self.trigger_mode_box.currentIndexChanged.connect(self.select_trigger_callback)

	def calc_trigger_wb(self):
		try:
			if self.trigger_mode == self.trig_modes[2]:
				self.trigger_frame = int(math.floor((self.end_frame-self.start_frame)/2.0)+1)
			elif self.trigger_mode == self.trig_modes[1]:
				self.trigger_frame = self.start_frame
			elif self.trigger_mode == self.trig_modes[3]:
				self.trigger_frame = self.end_frame
		except:
			print('error: could not calculate trigger frame')
			self.trigger_mode = self.trig_modes[0]
			self.trigger_mode_combo.setCurrentIndex(0)

	def select_trigger_callback(self,ind):
		self.trigger_mode = self.trig_modes[ind]
		self.calc_trigger_wb()

	def set_table_widget(self,table_in):
		self.par_table = table_in
		self.par_table.setRowCount(17)
		self.par_table.setColumnCount(7)
		self.par_table.setItem(0,0,QTableWidgetItem('Movie folders:'))
		self.par_table.setItem(0,1,QTableWidgetItem('Camera 1'))
		self.par_table.setItem(0,2,QTableWidgetItem('Camera 2'))
		self.par_table.setItem(0,3,QTableWidgetItem('Camera 3'))
		self.par_table.setItem(0,4,QTableWidgetItem('Camera 4'))
		self.par_table.setItem(0,5,QTableWidgetItem('Camera 5'))
		self.par_table.setItem(0,6,QTableWidgetItem('Camera 6'))
		self.par_table.resizeColumnsToContents()

	def update_table_widget(self):
		if self.N_mov > 0:
			for i in range(self.N_mov):
				self.par_table.setItem(i+1,0,QTableWidgetItem(self.mov_folder_list[i]))
				for j in range(self.N_cam):
					self.par_table.setItem(i+1,j+1,QTableWidgetItem(self.camera_folders[j]))
		self.par_table.resizeColumnsToContents()

	def load_seq_callback(self):
		self.active_frame_gui()

	#-----------------------------------------------------------------------
	#
	#	Background subtraction, calibration, masking and cropping
	#
	#-----------------------------------------------------------------------

	def inactive_frame_gui(self):
		# movie spin
		self.movie_spin_1.setMinimum(0)
		self.movie_spin_1.setMaximum(0)
		self.movie_spin_1.setValue(0)
		# frame spin
		self.frame_spin_1.setMinimum(0)
		self.frame_spin_1.setMaximum(0)
		self.frame_spin_1.setValue(0)
		# camera view spin 1
		self.camera_view_spin_1.setMinimum(0)
		self.camera_view_spin_1.setMaximum(0)
		self.camera_view_spin_1.setValue(0)
		# mask spin
		self.mask_spin.setMinimum(0)
		self.mask_spin.setMaximum(0)
		self.mask_spin.setValue(0)
		# camera view spin 2

	def active_frame_gui(self):
		# set sequence folder
		self.img_viewer_1.set_session_folder(self.session_path)
		self.img_viewer_1.set_N_cam(self.N_cam)
		self.img_viewer_1.set_trigger_mode(self.trigger_mode,self.trigger_frame)
		self.img_viewer_1.load_bckg_frames(self.bckg_path,self.bckg_frames,self.bckg_img_format)
		self.img_viewer_1.set_table_widget(self.img_center_table)
		# movie spin
		self.movie_spin_1.setMinimum(1)
		self.movie_spin_1.setMaximum(self.N_mov)
		self.movie_spin_1.setValue(1)
		self.img_viewer_1.set_movie_nr(1)
		self.movie_spin_1.valueChanged.connect(self.img_viewer_1.set_movie_nr)
		# frame spin
		self.frame_spin_1.setMinimum(self.start_frame)
		self.frame_spin_1.setMaximum(self.end_frame)
		self.frame_spin_1.setValue(self.start_frame)
		self.img_viewer_1.add_frame(self.start_frame)
		self.frame_spin_1.valueChanged.connect(self.img_viewer_1.update_frame)
		# segmentation view button
		#self.seg_view_btn.toggled.connect()
		# image view button
		#self.img_view_btn.toggled.connect()
		self.img_viewer_1.add_crop_graphs()
		self.img_viewer_1.setMouseCallbacks()
		self.set_img_cntr_btn.clicked.connect(self.set_crop_window)
		self.continue_1_btn.clicked.connect(self.annotation_gui)

	def set_crop_window(self):
		self.img_viewer_1.set_img_cntr()
		self.img_centers = self.img_viewer_1.uv_centers
		self.crop_window_size = self.img_viewer_1.crop_window
		self.thorax_center = self.img_viewer_1.thorax_center
		if self.annotation_active:
			self.img_viewer_2.load_crop_center(self.img_centers,self.crop_window_size,self.frame_size)

	def LoadCalibrationMatrix(self):
		os.chdir(self.calib_path)
		self.c_params = np.loadtxt(self.calib_file + '.txt', delimiter='\t')
		self.N_cam = self.c_params.shape[1]

	def CalculateProjectionMatrixes(self,pix_size,magnification):
		self.img_size = []
		self.w2c_matrices = []
		self.c2w_matrices = []
		self.uv_shift = []
		for i in range(self.N_cam):
			self.img_size.append((int(self.c_params[13,i]),int(self.c_params[12,i])))
			# Calculate world 2 camera transform:
			C = np.array([[self.c_params[0,i], self.c_params[2,i], 0.0, 0.0],
				[0.0, self.c_params[1,i], 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0]])
			q0 = self.c_params[3,i]
			q1 = self.c_params[4,i]
			q2 = self.c_params[5,i]
			q3 = self.c_params[6,i]
			R = np.array([[2.0*pow(q0,2)-1.0+2.0*pow(q1,2), 2.0*q1*q2+2.0*q0*q3,  2.0*q1*q3-2.0*q0*q2],
				[2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2)-1.0+2.0*pow(q2,2), 2.0*q2*q3+2.0*q0*q1],
				[2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2)-1.0+2.0*pow(q3,2)]])
			T = np.array([self.c_params[7,i],self.c_params[8,i],self.c_params[9,i]])
			K = np.array([[R[0,0], R[0,1], R[0,2], T[0]],
				[R[1,0], R[1,1], R[1,2], T[1]],
				[R[2,0], R[2,1], R[2,2], T[2]],
				[0.0, 0.0, 0.0, 1.0]])
			W2C_mat = np.dot(C,K)
			C2W_mat = np.dot(np.linalg.inv(K),np.linalg.pinv(C))
			self.w2c_matrices.append(W2C_mat)
			self.c2w_matrices.append(C2W_mat)

	def annotation_gui(self):
		self.annotation_active = True
		# setup img_viewer_2
		self.img_viewer_2.set_session_folder(self.session_path)
		self.img_viewer_2.set_N_cam(self.N_cam)
		self.img_viewer_2.set_output_folder('manual_tracking')
		self.img_viewer_2.set_trigger_mode(self.trigger_mode,self.trigger_frame)
		self.img_centers = self.img_viewer_1.uv_centers
		self.crop_window_size = self.img_viewer_1.crop_window
		self.img_viewer_2.load_camera_calibration(self.c_params,self.c2w_matrices,self.w2c_matrices)
		self.img_viewer_2.load_crop_center(self.img_centers,self.crop_window_size,self.frame_size,self.thorax_center)
		# activate tab
		self.tabWidget.setTabEnabled(1,True)
		# movie spin:
		self.movie_spin_2.setMinimum(1)
		self.movie_spin_2.setMaximum(self.N_mov)
		self.movie_spin_2.setValue(1)
		self.img_viewer_2.set_movie_nr(1)
		self.movie_spin_2.valueChanged.connect(self.img_viewer_2.set_movie_nr)
		# frame spin:
		self.frame_spin_2.setMinimum(self.start_frame)
		self.frame_spin_2.setMaximum(self.end_frame)
		self.frame_spin_2.setValue(self.start_frame)
		self.img_viewer_2.add_frame(self.start_frame)
		self.frame_spin_2.valueChanged.connect(self.img_viewer_2.update_frame)
		self.img_viewer_2.add_graphs()
		self.img_viewer_2.setMouseCallbacks()
		# network combo:
		self.network_combo.addItem(self.network_options[0])
		self.network_combo.addItem(self.network_options[1])
		self.network_combo.currentIndexChanged.connect(self.select_network)
		# select weight_file:
		self.select_weights_btn.clicked.connect(self.load_weights_callback)
		# predict state button:
		self.predict_state_btn.clicked.connect(self.predict_frame)
		# uv-shift:
		self.img_viewer_2.set_camera_view_spin(self.camera_view_spin_2)
		self.img_viewer_2.set_u_shift_spin(self.u_shift_spin)
		self.img_viewer_2.set_v_shift_spin(self.v_shift_spin)
		# Left wing:
		self.img_viewer_2.set_scale_L_spin(self.scale_L_spin)
		self.img_viewer_2.set_q1_L_spin(self.q1_L_spin)
		self.img_viewer_2.set_q2_L_spin(self.q2_L_spin)
		self.img_viewer_2.set_q3_L_spin(self.q3_L_spin)
		self.img_viewer_2.set_tx_L_spin(self.tx_L_spin)
		self.img_viewer_2.set_ty_L_spin(self.ty_L_spin)
		self.img_viewer_2.set_tz_L_spin(self.tz_L_spin)
		self.img_viewer_2.set_beta_L_spin(self.beta_L_spin)
		self.reset_L_btn.clicked.connect(self.img_viewer_2.reset_state_L)
		# Right wing:
		self.img_viewer_2.set_scale_R_spin(self.scale_R_spin)
		self.img_viewer_2.set_q1_R_spin(self.q1_R_spin)
		self.img_viewer_2.set_q2_R_spin(self.q2_R_spin)
		self.img_viewer_2.set_q3_R_spin(self.q3_R_spin)
		self.img_viewer_2.set_tx_R_spin(self.tx_R_spin)
		self.img_viewer_2.set_ty_R_spin(self.ty_R_spin)
		self.img_viewer_2.set_tz_R_spin(self.tz_R_spin)
		self.img_viewer_2.set_beta_R_spin(self.beta_R_spin)
		self.reset_R_btn.clicked.connect(self.img_viewer_2.reset_state_R)
		# create manual directory
		self.create_label_dir_btn.clicked.connect(self.img_viewer_2.create_manual_track_dir)
		# save label button
		self.save_label_btn.clicked.connect(self.img_viewer_2.save_frame)

	def select_network(self,net_ind):
		if net_ind == 1:
			print('selected network_27_11')
			os.chdir('/home/flyami/Documents/FlyNet/FlyNet_program/src')
			sys.path.append(os.getcwd())
			from network_27_11 import CNN_3D
			self.batch_size = 20
			self.net = CNN_3D(self.batch_size)
		else:
			print('no network selected')

	def load_weights_callback(self):
		self.select_weights_window.exec_()
		weight_file_loc = str(self.select_weights_window.folder_path)
		weight_file_name = str(self.select_weights_window.folder_name)
		if weight_file_name in weight_file_loc:
			self.weights_folder = weight_file_loc.replace(weight_file_name,'')
		else:
			self.weights_folder = weight_file_loc
		print(self.weights_folder)
		self.weights_file = weight_file_name
		self.weight_file_disp.setText(self.weights_file)
		print(self.weights_file)
		self.load_weights()

	def load_weights(self):
		os.chdir(self.weights_folder)
		if self.net:
			self.net.set_network_loc(self.weights_folder,self.weights_file)
			self.learning_rate = 0.0001
			self.decay = 0.00001
			self.net.set_learning_rate(self.learning_rate,self.decay)
			self.input_shape = (192,192,1,self.N_cam)
			self.output_dim = [120,18]
			self.net.load_network(self.input_shape,self.output_dim)
			print('weights loaded')
		else:
			print('Could not load weights, load network first.')

	def predict_frame(self):
		try:
			frame_list = self.img_viewer_2.load_frame()
			img_pred = np.zeros((1,self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3]))
			for n in range(self.N_cam):
				img_pred[0,:,:,0,n] = 1.0-frame_list[n]
			y_pred = self.net.predict_single_frame(img_pred)
			print(y_pred)
			# set scale and state:
			#if self.scale_set:
			#	self.state_L = y_pred[0,1:9]
			#	print(self.state_L)
			#	self.state_R = y_pred[0,10:18]
			#	print(self.state_R)
			#	self.img_viewer_2.set_state_L(y_pred[0,1:9])
			#	self.img_viewer_2.set_state_R(y_pred[0,10:18])
			#else:
			self.state_L = y_pred[0,1:9]
			self.state_R = y_pred[0,10:18]
			self.scale_L = y_pred[0,0]
			self.scale_R = y_pred[0,9]
			self.img_viewer_2.set_scale_L(y_pred[0,0])
			self.img_viewer_2.set_scale_R(y_pred[0,9])
			self.img_viewer_2.set_state_L(y_pred[0,1:9])
			self.img_viewer_2.set_state_R(y_pred[0,10:18])
		except:
			print('Could not predict state, load network first.')

# -------------------------------------------------------------------------------------------------

def appMain():
	app = QtWidgets.QApplication(sys.argv)
	mainWindow = FlyNetViewer()
	mainWindow.show()
	app.exec_()

# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	appMain()