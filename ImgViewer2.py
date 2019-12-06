from __future__ import print_function
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
import numpy as np
import numpy.matlib
import os
import os.path
from os import path
import copy
import time
import json
import h5py
import cv2

from state_fitter import StateFitter

class Graph(pg.GraphItem):
	def __init__(self,graph_nr):
		self.graph_nr = graph_nr
		self.dragPoint = None
		self.dragOffset = None
		self.textItems = []
		pg.GraphItem.__init__(self)
		self.scatter.sigClicked.connect(self.clicked)
		self.onMouseDragCb = None
		
	def setData(self, **kwds):
		self.text = kwds.pop('text', [])
		self.data = copy.deepcopy(kwds)
		
		if 'pos' in self.data:
			npts = self.data['pos'].shape[0]
			self.data['data'] = np.empty(npts, dtype=[('index', int)])
			self.data['data']['index'] = np.arange(npts)
		self.setTexts(self.text,self.data)
		self.updateGraph()
		
	def setTexts(self, text, data):
		for i in self.textItems:
			i.scene().removeItem(i)
		self.textItems = []
		#for t in text:
		for i,t in enumerate(text):
			item = pg.TextItem(t)
			if len(data.keys())>0:
				item.setColor(data['textcolor'][i])
			self.textItems.append(item)
			item.setParentItem(self)
		
	def updateGraph(self):
		pg.GraphItem.setData(self, **self.data)
		for i,item in enumerate(self.textItems):
			item.setPos(*self.data['pos'][i])

	def setOnMouseDragCallback(self, callback):
		self.onMouseDragCb = callback
		
	def mouseDragEvent(self, ev):
		if ev.button() != QtCore.Qt.LeftButton:
			ev.ignore()
			return
		
		if ev.isStart():
			# We are already one step into the drag.
			# Find the point(s) at the mouse cursor when the button was first 
			# pressed:
			pos = ev.buttonDownPos()
			pts = self.scatter.pointsAt(pos)
			if len(pts) == 0:
				ev.ignore()
				return
			self.dragPoint = pts[0]
			ind = pts[0].data()[0]
			self.dragOffset = self.data['pos'][ind] - pos
		elif ev.isFinish():
			self.dragPoint = None
			return
		else:
			if self.dragPoint is None:
				ev.ignore()
				return
		
		ind = self.dragPoint.data()[0]
		self.data['pos'][ind] = ev.pos() + self.dragOffset
		self.updateGraph()
		ev.accept()
		if self.onMouseDragCb:
			PosData = self.data['pos'][ind]
			PosData = np.append(PosData,ind)
			PosData = np.append(PosData,self.graph_nr)
			self.onMouseDragCb(PosData)
		
	def clicked(self, pts):
		print("clicked: %s" % pts)

class ImgViewer2(pg.GraphicsWindow):

	def __init__(self, parent=None):
		pg.GraphicsWindow.__init__(self)
		self.setParent(parent)

		self.w_sub = self.addLayout(row=0,col=0)

		self.v_list = []
		self.img_list = []
		self.frame_list = []

		# Parameters:
		self.frame_nr = 0
		self.N_cam = 3
		self.mov_folders = ['mov_1','mov_2','mov_3','mov_4','mov_5','mov_6','mov_7','mov_8','mov_9','mov_10']
		self.cam_folders = ['cam_1','cam_2','cam_3','cam_4','cam_5','cam_6','cam_7','cam_8','cam_9','cam_10']
		self.frame_name = 'frame_'
		self.trig_modes = ['start','center','end']
		self.graph_list = []

		# wing contours:
		self.state_calc = StateFitter()
		self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.scale_L = 1.0
		self.scale_R = 1.0
		self.key_pts_L = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, 2.6241, 0.0, 1.0]])
		self.wing_L_pts = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, 2.6241, 0.0, 1.0]])
		self.key_pts_R = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, -2.6241, 0.0, 1.0]])
		self.wing_R_pts = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, -2.6241, 0.0, 1.0]])
		self.wing_L_uv = []
		self.wing_R_uv = []
		self.drag_lines = True
		self.cnt_lines = False
		self.contours_L = []
		self.contours_R = []
		self.gain = 2.0
		self.ref_L = np.eye(3)
		self.q1_L = 0.0
		self.q2_L = 0.0
		self.q3_L = 0.0
		self.tx_L = 0.0
		self.ty_L = 0.0
		self.tz_L = 0.0
		self.beta_L = 0.0
		self.ref_R = np.eye(3)
		self.q1_R = 0.0
		self.q2_R = 0.0
		self.q3_R = 0.0
		self.tx_R = 0.0
		self.ty_R = 0.0
		self.tz_R = 0.0
		self.beta_R = 0.0

		# wing key points:
		self.wing_key_pts_L = np.transpose(np.array([
			[0.2313, 0.5711, 0.0, 1.0],
			[0.3253, 2.3205, 0.0, 1.0],
			[0.0, 2.6241, 0.0, 1.0],
			[-0.2386, 2.5591, 0.0, 1.0],
			[-0.7012, 1.5976, 0.0, 1.0],
			[-0.7880, 0.8892, 0.0, 1.0],
			[-0.4048, 1.2578, 0.0, 1.0],
			[-0.1952, 1.2868, 0.0, 1.0],
			[0.0072, 0.7157, 0.0, 1.0],
			[-0.0867, 0.0145, 0.0, 1.0]]))
		self.wing_key_pts_R = np.transpose(np.array([
			[0.2313, 0.5711, 0.0, 1.0],
			[0.3253, 2.3205, 0.0, 1.0],
			[0.0, 2.6241, 0.0, 1.0],
			[-0.2386, 2.5591, 0.0, 1.0],
			[-0.7012, 1.5976, 0.0, 1.0],
			[-0.7880, 0.8892, 0.0, 1.0],
			[-0.4048, 1.2578, 0.0, 1.0],
			[-0.1952, 1.2868, 0.0, 1.0],
			[0.0072, 0.7157, 0.0, 1.0],
			[-0.0867, 0.0145, 0.0, 1.0]]))

	def set_session_folder(self,session_folder):
		self.session_folder = session_folder

	def set_N_cam(self,N_cam):
		self.N_cam = N_cam

	def set_output_folder(self,output_folder):
		self.output_folder = output_folder

	def set_trigger_mode(self,trigger_mode,trigger_frame):
		self.trigger_mode = trigger_mode
		self.trigger_frame = trigger_frame

	def set_movie_nr(self,mov_nr):
		self.mov_nr = mov_nr-1
		self.mov_folder = self.mov_folders[self.mov_nr]

	def load_camera_calibration(self,c_params,c2w_matrices,w2c_matrices):
		self.c_params = c_params
		self.c2w_matrices = c2w_matrices
		self.w2c_matrices = w2c_matrices

	def set_camera_view_spin(self,spin_in):
		self.cam_view_spin = spin_in
		self.cam_view_spin.setMinimum(1)
		self.cam_view_spin.setMaximum(self.N_cam)
		self.cam_view_spin.setValue(1)
		self.set_camera_view(1)
		self.cam_view_spin.valueChanged.connect(self.set_camera_view)

	def set_camera_view(self,ind_in):
		self.cam_view = ind_in-1
		try:
			self.u_shift_spin.setValue(self.uv_shift[self.cam_view][0])
			self.v_shift_spin.setValue(self.uv_shift[self.cam_view][1])
		except:
			print(' ')

	def set_u_shift_spin(self,spin_in):
		self.u_shift_spin = spin_in
		self.u_shift_spin.setMinimum(-20)
		self.u_shift_spin.setMaximum(20)
		self.u_shift_spin.setValue(0)
		self.set_u_shift(0)
		self.u_shift_spin.valueChanged.connect(self.set_u_shift)

	def set_u_shift(self,u_in):
		self.uv_shift[self.cam_view][0] = u_in
		self.add_wing_contours()
		self.update_graphs()

	def set_v_shift_spin(self,spin_in):
		self.v_shift_spin = spin_in
		self.v_shift_spin.setMinimum(-20)
		self.v_shift_spin.setMaximum(20)
		self.v_shift_spin.setValue(0)
		self.set_v_shift(0)
		self.v_shift_spin.valueChanged.connect(self.set_v_shift)

	def set_v_shift(self,v_in):
		self.uv_shift[self.cam_view][1] = v_in
		self.add_wing_contours()
		self.update_graphs()

	def load_crop_center(self,img_centers,crop_window_size,frame_size,thorax_cntr):
		self.img_centers = []
		self.frame_size = frame_size
		self.crop_window = np.zeros((self.N_cam,8),dtype=int)
		self.crop_window_size = crop_window_size
		self.thorax_center = thorax_cntr
		print('thorax center img viewer 2')
		print(self.thorax_center)
		for n in range(self.N_cam):
			self.img_centers.append([int(img_centers[n][0]),int(img_centers[n][1])])
			# Calculate crop window dimensions:
			u_L = int(img_centers[n][0])-int(crop_window_size[n][0]/2.0)
			u_R = int(img_centers[n][0])+int(crop_window_size[n][0]/2.0)
			v_D = int(self.frame_size[n][1]-img_centers[n][1])-int(crop_window_size[n][1]/2.0)
			v_U = int(self.frame_size[n][1]-img_centers[n][1])+int(crop_window_size[n][1]/2.0)
			if u_L >= 0 and u_R < self.frame_size[n][0]:
				if v_D >= 0 and v_U < self.frame_size[n][1]:
					self.crop_window[n,0] = u_L
					self.crop_window[n,1] = v_D
					self.crop_window[n,2] = u_R
					self.crop_window[n,3] = v_U
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = 0
					self.crop_window[n,6] = crop_window_size[n][0]
					self.crop_window[n,7] = crop_window_size[n][1]
				elif v_D < 0:
					self.crop_window[n,0] = u_L
					self.crop_window[n,1] = 0
					self.crop_window[n,2] = u_R
					self.crop_window[n,3] = v_U
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = -v_D
					self.crop_window[n,6] = crop_window_size[n][0]
					self.crop_window[n,7] = crop_window_size[n][1]
				elif v_U >= self.frame_size[n][1]:
					self.crop_window[n,0] = u_L
					self.crop_window[n,1] = v_D
					self.crop_window[n,2] = u_R
					self.crop_window[n,3] = self.frame_size[n][1]
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = 0
					self.crop_window[n,6] = crop_window_size[n][0]
					self.crop_window[n,7] = crop_window_size[n][1]-(v_U-self.frame_size[n][1])
			elif u_L < 0:
				if v_D >= 0 and v_U < self.frame_size[n][1]:
					self.crop_window[n,0] = 0
					self.crop_window[n,1] = v_D
					self.crop_window[n,2] = u_R
					self.crop_window[n,3] = v_U
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = 0
					self.crop_window[n,6] = crop_window_size[n][0]
					self.crop_window[n,7] = crop_window_size[n][1]
				elif v_D < 0:
					self.crop_window[n,0] = 0
					self.crop_window[n,1] = 0
					self.crop_window[n,2] = u_R
					self.crop_window[n,3] = v_U
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = -v_D
					self.crop_window[n,6] = crop_window_size[n][0]
					self.crop_window[n,7] = crop_window_size[n][1]
				elif v_U >= self.frame_size[n][1]:
					self.crop_window[n,0] = 0
					self.crop_window[n,1] = v_D
					self.crop_window[n,2] = u_R
					self.crop_window[n,3] = self.frame_size[n][1]
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = 0
					self.crop_window[n,6] = crop_window_size[n][0]
					self.crop_window[n,7] = crop_window_size[n][1]-(v_U-self.frame_size[n][1])
			elif u_R >=  self.frame_size[n][0]:
				if v_D >= 0 and v_U < self.frame_size[n][1]:
					self.crop_window[n,0] = u_L
					self.crop_window[n,1] = v_D
					self.crop_window[n,2] = self.frame_size[n][0]
					self.crop_window[n,3] = v_U
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = 0
					self.crop_window[n,6] = crop_window_size[n][0]-(u_R-self.frame_size[n][0])
					self.crop_window[n,7] = crop_window_size[n][1]
				elif v_D < 0:
					self.crop_window[n,0] = u_L
					self.crop_window[n,1] = 0
					self.crop_window[n,2] = self.frame_size[n][0]
					self.crop_window[n,3] = v_U
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = -v_D
					self.crop_window[n,6] = crop_window_size[n][0]-(u_R-self.frame_size[n][0])
					self.crop_window[n,7] = crop_window_size[n][1]
				elif v_U >= self.frame_size[n][1]:
					self.crop_window[n,0] = u_L
					self.crop_window[n,1] = v_D
					self.crop_window[n,2] = self.frame_size[n][0]
					self.crop_window[n,3] = self.frame_size[n][1]
					self.crop_window[n,4] = 0
					self.crop_window[n,5] = 0
					self.crop_window[n,6] = crop_window_size[n][0]-(u_R-self.frame_size[n][0])
					self.crop_window[n,7] = crop_window_size[n][1]-(v_U-self.frame_size[n][1])
		self.uv_offset = []
		self.uv_shift = []
		for n in range(self.N_cam):
			#uv_thorax = np.dot(self.w2c_matrices[n],self.thorax_center)
			uv_thorax = np.dot(self.w2c_matrices[n],np.array([[0.0],[0.0],[0.0],[1.0]]))
			uv_trans = np.zeros((3,1))
			uv_trans[0] = uv_thorax[0]-crop_window_size[n][0]/2.0 #+self.crop_window[n,0] #-(self.c_params[11,n]-self.frame_size[n][0])/2.0-self.crop_window[n,0]
			uv_trans[1] = uv_thorax[1]-crop_window_size[n][1]/2.0 #+self.crop_window[n,1] #-(self.c_params[10,n]-self.frame_size[n][1])/2.0-self.crop_window[n,1]
			self.uv_offset.append(uv_trans)
			self.uv_shift.append(np.zeros((3,1)))

	def load_frame(self):
		frame_list = []
		for i in range(self.N_cam):
			if self.trigger_mode == 'start':
				frame_ind = self.frame_nr
			elif self.trigger_mode == 'center':
				if self.frame_nr < self.trigger_frame:
					frame_ind = self.frame_nr+self.trigger_frame
				else:
					frame_ind = self.frame_nr-self.trigger_frame
			elif self.trigger_mode == 'end':
				frame_ind = self.frame_nr
			else:
				print('error: invalid trigger mode')
			os.chdir(self.session_folder+'/'+self.mov_folder)
			os.chdir(self.cam_folders[i])
			img_cv = cv2.imread(self.frame_name + str(frame_ind) +'.bmp',0)
			img_cv = img_cv/255.0
			# Crop window:
			img_cropped = np.ones((self.crop_window_size[i][0],self.crop_window_size[i][1]))
			img_cropped[self.crop_window[i,5]:self.crop_window[i,7],self.crop_window[i,4]:self.crop_window[i,6]] = img_cv[self.crop_window[i,1]:self.crop_window[i,3],self.crop_window[i,0]:self.crop_window[i,2]]
			frame_list.append(img_cropped)
		return frame_list

	def add_frame(self,frame_nr):
		self.frame_nr = frame_nr
		frame_list = self.load_frame()
		print(frame_list[0].shape)
		for i, frame in enumerate(frame_list):
			self.frame_list.append(frame)
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			frame_in = np.transpose(np.flipud(frame))
			self.img_list.append(pg.ImageItem(frame_in))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()

	def update_frame(self,frame_nr):
		self.frame_nr = frame_nr
		frame_list = self.load_frame()
		for i, frame in enumerate(frame_list):
			frame_in = np.transpose(np.flipud(frame))
			self.img_list[i].setImage(frame_in)

	def project2uv(self):
		self.wing_L_uv = []
		self.wing_R_uv = []
		for n in range(self.N_cam):
			uv_L_pts = np.dot(self.w2c_matrices[n],np.transpose(self.wing_L_pts))-self.uv_offset[n]-self.uv_shift[n]
			uv_L_pts[1,:] = self.crop_window_size[n][1]-uv_L_pts[1,:]
			uv_R_pts = np.dot(self.w2c_matrices[n],np.transpose(self.wing_R_pts))-self.uv_offset[n]-self.uv_shift[n]
			uv_R_pts[1,:] = self.crop_window_size[n][1]-uv_R_pts[1,:]
			self.wing_L_uv.append(uv_L_pts[0:2,:])
			self.wing_R_uv.append(uv_R_pts[0:2,:])

	def add_graphs(self):
		self.graph_list = []
		self.project2uv()
		for i in range(self.N_cam):
			self.graph_list.append(Graph(i))
			self.v_list[i].addItem(self.graph_list[i])
			self.wing_txt = ['L_r','L_t','R_r','R_t']
			self.wing_sym = ["o","o","o","o"]
			self.wing_clr = ['r','r','b','b']
			uv_pos = np.concatenate((np.transpose(self.wing_L_uv[i]),np.transpose(self.wing_R_uv[i])),axis=0)
			print(uv_pos)
			self.graph_list[i].setData(pos=uv_pos, size=2, symbol=self.wing_sym, pxMode=False, text=self.wing_txt, textcolor=self.wing_clr)
		self.add_wing_contours()

	def update_graphs(self):
		# Update keypoints:
		M_now_L = self.state_transform_L()
		key_pts_L0 = np.transpose(self.key_pts_L)
		self.wing_L_pts[0:3,:] = np.transpose(np.dot(M_now_L[0],key_pts_L0))
		ref_frame_L = np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]])
		self.ref_L = np.dot(M_now_L[0],ref_frame_L)
		M_now_R = self.state_transform_R()
		key_pts_R0 = np.transpose(self.key_pts_R)
		self.wing_R_pts[0:3,:] = np.transpose(np.dot(M_now_R[0],key_pts_R0))
		ref_frame_R = np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]])
		self.ref_R = np.dot(M_now_R[0],ref_frame_R)
		# Project 2 UV
		self.project2uv()
		for i in range(self.N_cam):
			uv_pos = np.concatenate((np.transpose(self.wing_L_uv[i]),np.transpose(self.wing_R_uv[i])),axis=0)
			self.graph_list[i].setData(pos=uv_pos, size=2, symbol=self.wing_sym, pxMode=False, text=self.wing_txt, textcolor=self.wing_clr)

	def remove_graphs(self):
		if len(self.graph_list)>0:
			for i in range(self.N_cam):
				self.v_list[i].removeItem(self.graph_list[i])
			self.graph_list = []

	def contours2uv(self,pts_in):
		cnt_uv = []
		for n in range(self.N_cam):
			uv_pts = np.dot(self.w2c_matrices[n],pts_in)-self.uv_offset[n]-self.uv_shift[n]
			uv_pts[1,:] = self.crop_window_size[n][1]-uv_pts[1,:]
			cnt_uv.append(uv_pts)
		return cnt_uv

	def add_wing_contours(self):
		self.remove_wing_contours()
		# Update state
		self.state_calc.set_state(self.state_L,self.state_R)
		# Retrieve 3d coordinates left and right wings:
		wing_L_cnts = self.state_calc.wing_contour_L()
		wing_R_cnts = self.state_calc.wing_contour_R()
		# obtain 2D projections:
		cnts_L_uv = []
		for cnt in wing_L_cnts:
			cnts_L_uv.append(self.contours2uv(cnt))
		cnts_R_uv = []
		for cnt in wing_R_cnts:
			cnts_R_uv.append(self.contours2uv(cnt))
		# Add contour plots to the image items:
		self.contours_L = []
		for i,cnt_pts in enumerate(cnts_L_uv):
			for n in range(self.N_cam):
				curve_pts = np.transpose(cnt_pts[n][0:2,:])
				curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[255,0,0])
				self.contours_L.append(curve)
				self.v_list[n].addItem(self.contours_L[i*self.N_cam+n])
		self.contours_R = []
		for i,cnt_pts in enumerate(cnts_R_uv):
			for n in range(self.N_cam):
				curve_pts = np.transpose(cnt_pts[n][0:2,:])
				curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[0,0,255])
				self.contours_R.append(curve)
				self.v_list[n].addItem(self.contours_R[i*self.N_cam+n])

	def remove_wing_contours(self):
		N_L = len(self.contours_L)
		if N_L>0:
			for i in range(N_L):
				self.v_list[i%self.N_cam].removeItem(self.contours_L[i])
			self.contours_L = []
		N_R = len(self.contours_R)
		if N_R>0:
			for i in range(N_R):
				self.v_list[i%self.N_cam].removeItem(self.contours_R[i])
			self.contours_R = []

	def state_transform_L(self):
		q_norm = np.sqrt(pow(self.state_L[0],2)+pow(self.state_L[1],2)+pow(self.state_L[2],2)+pow(self.state_L[3],2))
		q_0 = np.array([[self.state_L[0]],
			[self.state_L[1]],
			[self.state_L[2]],
			[self.state_L[3]]])/q_norm
		T = np.array([
			[self.state_L[4]],
			[self.state_L[5]],
			[self.state_L[6]]])
		b1 = self.state_L[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = self.scale_L*np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
			[2.0*q_0[1]*q_0[2]-2.0*q_0[0]*q_0[3], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[2],2), 2.0*q_0[2]*q_0[3]+2.0*q_0[0]*q_0[1]],
			[2.0*q_0[1]*q_0[3]+2.0*q_0[0]*q_0[2], 2.0*q_0[2]*q_0[3]-2.0*q_0[0]*q_0[1], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[3],2)]])
		q_1 = np.array([
			[np.cos(b1/2.0)],
			[0.0],
			[np.sin(b1/2.0)],
			[0.0]])
		R_1 = np.array([[2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[1],2), 2.0*q_1[1]*q_1[2]+2.0*q_1[0]*q_1[3],  2.0*q_1[1]*q_1[3]-2.0*q_1[0]*q_1[2]],
			[2.0*q_1[1]*q_1[2]-2.0*q_1[0]*q_1[3], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[2],2), 2.0*q_1[2]*q_1[3]+2.0*q_1[0]*q_1[1]],
			[2.0*q_1[1]*q_1[3]+2.0*q_1[0]*q_1[2], 2.0*q_1[2]*q_1[3]-2.0*q_1[0]*q_1[1], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[3],2)]])
		q_2 = np.array([
			[np.cos(b2/2.0)],
			[-0.05959*np.sin(b2/2.0)],
			[0.99822*np.sin(b2/2.0)],
			[0.0]])
		R_2 = np.array([[2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[1],2), 2.0*q_2[1]*q_2[2]+2.0*q_2[0]*q_2[3],  2.0*q_2[1]*q_2[3]-2.0*q_2[0]*q_2[2]],
			[2.0*q_2[1]*q_2[2]-2.0*q_2[0]*q_2[3], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[2],2), 2.0*q_2[2]*q_2[3]+2.0*q_2[0]*q_2[1]],
			[2.0*q_2[1]*q_2[3]+2.0*q_2[0]*q_2[2], 2.0*q_2[2]*q_2[3]-2.0*q_2[0]*q_2[1], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[3],2)]])
		q_3 = np.array([
			[np.cos(b3/2.0)],
			[-0.36186*np.sin(b3/2.0)],
			[0.93223*np.sin(b3/2.0)],
			[0.0]])
		R_3 = np.array([[2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[1],2), 2.0*q_3[1]*q_3[2]+2.0*q_3[0]*q_3[3],  2.0*q_3[1]*q_3[3]-2.0*q_3[0]*q_3[2]],
			[2.0*q_3[1]*q_3[2]-2.0*q_3[0]*q_3[3], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[2],2), 2.0*q_3[2]*q_3[3]+2.0*q_3[0]*q_3[1]],
			[2.0*q_3[1]*q_3[3]+2.0*q_3[0]*q_3[2], 2.0*q_3[2]*q_3[3]-2.0*q_3[0]*q_3[1], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[3],2)]])
		# transform key_pts_0:
		M_0 = np.zeros((4,4))
		M_0[0:3,0:3] = np.squeeze(R_0)
		M_0[0:3,3] = np.squeeze(T)
		M_0[3,3] = 1.0
		# transform key_pts_1:
		M_1 = np.zeros((4,4))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		# transform key_pts_2:
		M_2 = np.zeros((4,4))
		M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		# transform key_pts_3:
		M_3 = np.zeros((4,4))
		M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		# Return list of transformations:
		M_list = [M_0, M_1, M_2, M_3]
		return M_list

	def state_transform_R(self):
		q_norm = np.sqrt(pow(self.state_R[0],2)+pow(self.state_R[1],2)+pow(self.state_R[2],2)+pow(self.state_R[3],2))
		q_0 = np.array([[self.state_R[0]],
			[self.state_R[1]],
			[self.state_R[2]],
			[self.state_R[3]]])/q_norm
		T = np.array([
			[self.state_R[4]],
			[self.state_R[5]],
			[self.state_R[6]]])
		b1 = self.state_R[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = self.scale_R*np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
			[2.0*q_0[1]*q_0[2]-2.0*q_0[0]*q_0[3], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[2],2), 2.0*q_0[2]*q_0[3]+2.0*q_0[0]*q_0[1]],
			[2.0*q_0[1]*q_0[3]+2.0*q_0[0]*q_0[2], 2.0*q_0[2]*q_0[3]-2.0*q_0[0]*q_0[1], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[3],2)]])
		q_1 = np.array([
			[np.cos(b1/2.0)],
			[0.0],
			[np.sin(b1/2.0)],
			[0.0]])
		R_1 = np.array([[2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[1],2), 2.0*q_1[1]*q_1[2]+2.0*q_1[0]*q_1[3],  2.0*q_1[1]*q_1[3]-2.0*q_1[0]*q_1[2]],
			[2.0*q_1[1]*q_1[2]-2.0*q_1[0]*q_1[3], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[2],2), 2.0*q_1[2]*q_1[3]+2.0*q_1[0]*q_1[1]],
			[2.0*q_1[1]*q_1[3]+2.0*q_1[0]*q_1[2], 2.0*q_1[2]*q_1[3]-2.0*q_1[0]*q_1[1], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[3],2)]])
		q_2 = np.array([
			[np.cos(b2/2.0)],
			[0.05959*np.sin(b2/2.0)],
			[0.99822*np.sin(b2/2.0)],
			[0.0]])
		R_2 = np.array([[2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[1],2), 2.0*q_2[1]*q_2[2]+2.0*q_2[0]*q_2[3],  2.0*q_2[1]*q_2[3]-2.0*q_2[0]*q_2[2]],
			[2.0*q_2[1]*q_2[2]-2.0*q_2[0]*q_2[3], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[2],2), 2.0*q_2[2]*q_2[3]+2.0*q_2[0]*q_2[1]],
			[2.0*q_2[1]*q_2[3]+2.0*q_2[0]*q_2[2], 2.0*q_2[2]*q_2[3]-2.0*q_2[0]*q_2[1], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[3],2)]])
		q_3 = np.array([
			[np.cos(b3/2.0)],
			[0.36186*np.sin(b3/2.0)],
			[0.93223*np.sin(b3/2.0)],
			[0.0]])
		R_3 = np.array([[2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[1],2), 2.0*q_3[1]*q_3[2]+2.0*q_3[0]*q_3[3],  2.0*q_3[1]*q_3[3]-2.0*q_3[0]*q_3[2]],
			[2.0*q_3[1]*q_3[2]-2.0*q_3[0]*q_3[3], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[2],2), 2.0*q_3[2]*q_3[3]+2.0*q_3[0]*q_3[1]],
			[2.0*q_3[1]*q_3[3]+2.0*q_3[0]*q_3[2], 2.0*q_3[2]*q_3[3]-2.0*q_3[0]*q_3[1], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[3],2)]])
		# transform key_pts_0:
		M_0 = np.zeros((4,4))
		M_0[0:3,0:3] = np.squeeze(R_0)
		M_0[0:3,3] = np.squeeze(T)
		M_0[3,3] = 1.0
		# transform key_pts_1:
		M_1 = np.zeros((4,4))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		# transform key_pts_2:
		M_2 = np.zeros((4,4))
		M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		# transform key_pts_3:
		M_3 = np.zeros((4,4))
		M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		# Return list of transformations:
		M_list = [M_0, M_1, M_2, M_3]
		return M_list

	def quat_multiply(self,qA,qB):
		QA = np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
			[qA[1],qA[0],-qA[3],qA[2]],
			[qA[2],qA[3],qA[0],-qA[1]],
			[qA[3],-qA[2],qA[1],qA[0]]])
		qc = np.dot(QA,qB)
		qc = qc/np.linalg.norm(qc)
		return qc

	def update_3D_points(self,u_prev,v_prev,u_now,v_now,point_nr,cam_nr):
		xyz_prev = np.squeeze(np.dot(self.c2w_matrices[cam_nr],np.array([[u_prev-self.uv_offset[cam_nr][0]-self.uv_shift[cam_nr][0]],[self.c_params[10,cam_nr]-v_prev+self.uv_offset[cam_nr][0]+self.uv_shift[cam_nr][0]],[0.0]])))
		xyz_now = np.squeeze(np.dot(self.c2w_matrices[cam_nr],np.array([[u_now-self.uv_offset[cam_nr][0]-self.uv_shift[cam_nr][0]],[self.c_params[10,cam_nr]-v_now+self.uv_offset[cam_nr][0]+self.uv_shift[cam_nr][0]],[0.0]])))
		d_xyz = xyz_now-xyz_prev
		if point_nr == 0:
			# translation L0
			self.state_L[4] = self.state_L[4]+d_xyz[0]
			self.tx_L_spin.setValue(np.around(self.state_L[4],decimals=3))
			self.set_tx_L(np.around(self.state_L[4],decimals=3))
			self.state_L[5] = self.state_L[5]+d_xyz[1]
			self.ty_L_spin.setValue(np.around(self.state_L[5],decimals=3))
			self.set_ty_L(np.around(self.state_L[5],decimals=3))
			self.state_L[6] = self.state_L[6]+d_xyz[2]
			self.tz_L_spin.setValue(np.around(self.state_L[6],decimals=3))
			self.set_tz_L(np.around(self.state_L[6],decimals=3))
		elif point_nr == 1:
			q_prev = self.state_L[0:4]
			M_list_L = self.state_transform_L()
			rot_vec = np.dot(np.transpose(M_list_L[0]),d_xyz)
			delta_q1 = np.squeeze(np.arctan2(rot_vec[2],self.key_pts_L[1,1]*self.scale_L+rot_vec[1]))/(2.0*np.pi)
			delta_q3 = -np.squeeze(np.arctan2(rot_vec[0],self.key_pts_L[1,1]*self.scale_L+rot_vec[1]))/(2.0*np.pi)
			self.set_q1_L(self.q1_L+delta_q1)
			self.q1_L_spin.setValue(self.q1_L+delta_q1)
			self.set_q3_L(self.q3_L+delta_q3)
			self.q3_L_spin.setValue(self.q3_L+delta_q3)
		elif point_nr == 2:
			# translation R0
			self.state_R[4] = self.state_R[4]+d_xyz[0]
			self.tx_R_spin.setValue(np.around(self.state_R[4],decimals=3))
			self.set_tx_R(np.around(self.state_R[4],decimals=3))
			self.state_R[5] = self.state_R[5]+d_xyz[1]
			self.ty_R_spin.setValue(np.around(self.state_R[5],decimals=3))
			self.set_ty_R(np.around(self.state_R[5],decimals=3))
			self.state_R[6] = self.state_R[6]+d_xyz[2]
			self.tz_R_spin.setValue(np.around(self.state_R[6],decimals=3))
			self.set_tz_R(np.around(self.state_R[6],decimals=3))
		elif point_nr == 3:
			q_prev = self.state_R[0:4]
			M_list_R = self.state_transform_R()
			rot_vec = np.dot(np.transpose(M_list_R[0]),d_xyz)
			delta_q1 = -np.squeeze(np.arctan2(rot_vec[2],self.key_pts_L[1,1]*self.scale_L+rot_vec[1]))/(2.0*np.pi)
			delta_q3 = np.squeeze(np.arctan2(rot_vec[0],self.key_pts_L[1,1]*self.scale_L+rot_vec[1]))/(2.0*np.pi)
			self.set_q1_R(self.q1_R+delta_q1)
			self.q1_R_spin.setValue(self.q1_R+delta_q1)
			self.set_q3_R(self.q3_R+delta_q3)
			self.q3_R_spin.setValue(self.q3_R+delta_q3)
		self.add_wing_contours()

	def setMouseCallbacks(self):
		def onMouseDragCallback(data):
			cam_nr = int(data[3])
			point_nr = int(data[2])
			if point_nr<2:
				u_prev = self.wing_L_uv[cam_nr][0,point_nr]
				v_prev = self.wing_L_uv[cam_nr][1,point_nr]
				#print('uv_prev: ' + str(u_prev) + ', ' + str(v_prev))
				u_now = data[0]
				v_now = data[1]
				#print('uv_now: ' + str(u_now) + ', ' + str(v_now))
				self.wing_L_uv[cam_nr][0,point_nr] = u_now
				self.wing_L_uv[cam_nr][1,point_nr] = v_now
			else:
				u_prev = self.wing_R_uv[cam_nr][0,point_nr-2]
				v_prev = self.wing_R_uv[cam_nr][1,point_nr-2]
				#print('uv_prev: ' + str(u_prev) + ', ' + str(v_prev))
				u_now = data[0]
				v_now = data[1]
				#print('uv_now: ' + str(u_now) + ', ' + str(v_now))
				self.wing_R_uv[cam_nr][0,point_nr-2] = u_now
				self.wing_R_uv[cam_nr][1,point_nr-2] = v_now
			self.update_3D_points(u_prev,v_prev,u_now,v_now,point_nr,cam_nr)
			self.update_graphs()

		for i in range(self.N_cam):
			self.graph_list[i].setOnMouseDragCallback(onMouseDragCallback)

	def set_state_L(self,state_in):
		self.state_L = state_in
		self.add_wing_contours()
		self.update_graphs()

	def set_state_R(self,state_in):
		self.state_R = state_in
		self.add_wing_contours()
		self.update_graphs()

	def set_scale_L_spin(self,spin_in):
		self.scale_L_spin = spin_in
		self.scale_L_spin.setMinimum(0.1)
		self.scale_L_spin.setMaximum(2.0)
		self.scale_L_spin.setDecimals(2)
		self.scale_L_spin.setSingleStep(0.01)
		self.scale_L_spin.setValue(1.0)
		self.set_scale_L(1.0)
		self.scale_L_spin.valueChanged.connect(self.set_scale_L)

	def set_scale_L(self,scale_in):
		self.scale_L = scale_in
		self.state_calc.set_scale(self.scale_L,self.scale_R)
		self.add_wing_contours()
		self.update_graphs()

	def set_q1_L_spin(self,spin_in):
		self.q1_L_spin = spin_in
		self.q1_L_spin.setMinimum(-2.0)
		self.q1_L_spin.setMaximum(2.0)
		self.q1_L_spin.setDecimals(3)
		self.q1_L_spin.setSingleStep(0.001)
		self.q1_L_spin.setValue(0.0)
		self.set_q1_L(0.0)
		self.q1_L_spin.valueChanged.connect(self.set_q1_L)

	def set_q1_L(self,q1_in):
		delta_q = np.pi*(self.q1_L-q1_in)
		self.q1_L = q1_in
		e_rot = self.ref_L[:,0]
		q_prev = self.state_L[0:4]
		q_update = np.array([np.cos(delta_q/2.0),e_rot[0]*np.sin(delta_q/2.0),e_rot[1]*np.sin(delta_q/2.0),e_rot[2]*np.sin(delta_q/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_L[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()

	def set_q2_L_spin(self,spin_in):
		self.q2_L_spin = spin_in
		self.q2_L_spin.setMinimum(-2.0)
		self.q2_L_spin.setMaximum(2.0)
		self.q2_L_spin.setDecimals(3)
		self.q2_L_spin.setSingleStep(0.001)
		self.q2_L_spin.setValue(0.0)
		self.set_q2_L(0.0)
		self.q2_L_spin.valueChanged.connect(self.set_q2_L)

	def set_q2_L(self,q2_in):
		delta_q = np.pi*(self.q2_L-q2_in)
		self.q2_L = q2_in
		e_rot = self.ref_L[:,1]
		q_prev = self.state_L[0:4]
		q_update = np.array([np.cos(delta_q/2.0),e_rot[0]*np.sin(delta_q/2.0),e_rot[1]*np.sin(delta_q/2.0),e_rot[2]*np.sin(delta_q/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_L[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()

	def set_q3_L_spin(self,spin_in):
		self.q3_L_spin = spin_in
		self.q3_L_spin.setMinimum(-2.0)
		self.q3_L_spin.setMaximum(2.0)
		self.q3_L_spin.setDecimals(3)
		self.q3_L_spin.setSingleStep(0.001)
		self.q3_L_spin.setValue(0.0)
		self.set_q3_L(0.0)
		self.q3_L_spin.valueChanged.connect(self.set_q3_L)

	def set_q3_L(self,q3_in):
		delta_q = np.pi*(self.q3_L-q3_in)
		self.q3_L = q3_in
		e_rot = self.ref_L[:,2]
		q_prev = self.state_L[0:4]
		q_update = np.array([np.cos(delta_q/2.0),e_rot[0]*np.sin(delta_q/2.0),e_rot[1]*np.sin(delta_q/2.0),e_rot[2]*np.sin(delta_q/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_L[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()

	def set_tx_L_spin(self,spin_in):
		self.tx_L_spin = spin_in
		self.tx_L_spin.setMinimum(-10.0)
		self.tx_L_spin.setMaximum(10.0)
		self.tx_L_spin.setDecimals(3)
		self.tx_L_spin.setSingleStep(0.001)
		self.tx_L_spin.setValue(0.0)
		self.set_tx_L(0.0)
		self.tx_L_spin.valueChanged.connect(self.set_tx_L)

	def set_tx_L(self,tx_in):
		self.tx_L = tx_in
		self.state_L[4] = tx_in
		self.add_wing_contours()
		self.update_graphs()

	def set_ty_L_spin(self,spin_in):
		self.ty_L_spin = spin_in
		self.ty_L_spin.setMinimum(-10.0)
		self.ty_L_spin.setMaximum(10.0)
		self.ty_L_spin.setDecimals(3)
		self.ty_L_spin.setSingleStep(0.001)
		self.ty_L_spin.setValue(0.0)
		self.set_ty_L(0.0)
		self.ty_L_spin.valueChanged.connect(self.set_ty_L)

	def set_ty_L(self,ty_in):
		self.ty_L = ty_in
		self.state_L[5] = ty_in
		self.add_wing_contours()
		self.update_graphs()

	def set_tz_L_spin(self,spin_in):
		self.tz_L_spin = spin_in
		self.tz_L_spin.setMinimum(-10.0)
		self.tz_L_spin.setMaximum(10.0)
		self.tz_L_spin.setDecimals(3)
		self.tz_L_spin.setSingleStep(0.001)
		self.tz_L_spin.setValue(0.0)
		self.set_tz_L(0.0)
		self.tz_L_spin.valueChanged.connect(self.set_tz_L)

	def set_tz_L(self,tz_in):
		self.tz_L = tz_in
		self.state_L[6] = tz_in
		self.add_wing_contours()
		self.update_graphs()

	def set_beta_L_spin(self,spin_in):
		self.beta_L_spin = spin_in
		self.beta_L_spin.setMinimum(-1.0)
		self.beta_L_spin.setMaximum(1.0)
		self.beta_L_spin.setDecimals(3)
		self.beta_L_spin.setSingleStep(0.001)
		self.beta_L_spin.setValue(0.0)
		self.set_beta_L(0.0)
		self.beta_L_spin.valueChanged.connect(self.set_beta_L)

	def set_beta_L(self,beta_in):
		self.state_L[7] = np.pi*beta_in
		self.add_wing_contours()
		self.update_graphs()

	def reset_state_L(self):
		self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.set_q1_L(0.0)
		self.q1_L_spin.setValue(0.0)
		self.set_q2_L(0.0)
		self.q2_L_spin.setValue(0.0)
		self.set_q3_L(0.0)
		self.q3_L_spin.setValue(0.0)
		self.set_tx_L(0.0)
		self.tx_L_spin.setValue(0.0)
		self.set_ty_L(0.0)
		self.ty_L_spin.setValue(0.0)
		self.set_tz_L(0.0)
		self.tz_L_spin.setValue(0.0)
		self.set_beta_L(0.0)
		self.beta_L_spin.setValue(0.0)

	def set_scale_R_spin(self,spin_in):
		self.scale_R_spin = spin_in
		self.scale_R_spin.setMinimum(0.1)
		self.scale_R_spin.setMaximum(2.0)
		self.scale_R_spin.setDecimals(2)
		self.scale_R_spin.setSingleStep(0.01)
		self.scale_R_spin.setValue(1.0)
		self.set_scale_R(1.0)
		self.scale_R_spin.valueChanged.connect(self.set_scale_R)

	def set_scale_R(self,scale_in):
		self.scale_R = scale_in
		self.state_calc.set_scale(self.scale_L,self.scale_R)
		self.add_wing_contours()
		self.update_graphs()

	def set_q1_R_spin(self,spin_in):
		self.q1_R_spin = spin_in
		self.q1_R_spin.setMinimum(-2.0)
		self.q1_R_spin.setMaximum(2.0)
		self.q1_R_spin.setDecimals(3)
		self.q1_R_spin.setSingleStep(0.001)
		self.q1_R_spin.setValue(0.0)
		self.set_q1_R(0.0)
		self.q1_R_spin.valueChanged.connect(self.set_q1_R)

	def set_q1_R(self,q1_in):
		delta_q = np.pi*(self.q1_R-q1_in)
		self.q1_R = q1_in
		e_rot = self.ref_R[:,0]
		q_prev = self.state_R[0:4]
		q_update = np.array([np.cos(delta_q/2.0),e_rot[0]*np.sin(delta_q/2.0),e_rot[1]*np.sin(delta_q/2.0),e_rot[2]*np.sin(delta_q/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_R[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()

	def set_q2_R_spin(self,spin_in):
		self.q2_R_spin = spin_in
		self.q2_R_spin.setMinimum(-2.0)
		self.q2_R_spin.setMaximum(2.0)
		self.q2_R_spin.setDecimals(3)
		self.q2_R_spin.setSingleStep(0.001)
		self.q2_R_spin.setValue(0.0)
		self.set_q2_R(0.0)
		self.q2_R_spin.valueChanged.connect(self.set_q2_R)

	def set_q2_R(self,q2_in):
		delta_q = np.pi*(self.q2_R-q2_in)
		self.q2_R = q2_in
		e_rot = self.ref_R[:,1]
		q_prev = self.state_R[0:4]
		q_update = np.array([np.cos(delta_q/2.0),e_rot[0]*np.sin(delta_q/2.0),e_rot[1]*np.sin(delta_q/2.0),e_rot[2]*np.sin(delta_q/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_R[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()

	def set_q3_R_spin(self,spin_in):
		self.q3_R_spin = spin_in
		self.q3_R_spin.setMinimum(-2.0)
		self.q3_R_spin.setMaximum(2.0)
		self.q3_R_spin.setDecimals(3)
		self.q3_R_spin.setSingleStep(0.001)
		self.q3_R_spin.setValue(0.0)
		self.set_q3_R(0.0)
		self.q3_R_spin.valueChanged.connect(self.set_q3_R)

	def set_q3_R(self,q3_in):
		delta_q = np.pi*(self.q3_R-q3_in)
		self.q3_R = q3_in
		e_rot = self.ref_R[:,2]
		q_prev = self.state_R[0:4]
		q_update = np.array([np.cos(delta_q/2.0),e_rot[0]*np.sin(delta_q/2.0),e_rot[1]*np.sin(delta_q/2.0),e_rot[2]*np.sin(delta_q/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_R[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()

	def set_tx_R_spin(self,spin_in):
		self.tx_R_spin = spin_in
		self.tx_R_spin.setMinimum(-10.0)
		self.tx_R_spin.setMaximum(10.0)
		self.tx_R_spin.setDecimals(3)
		self.tx_R_spin.setSingleStep(0.001)
		self.tx_R_spin.setValue(0.0)
		self.set_tx_R(0.0)
		self.tx_R_spin.valueChanged.connect(self.set_tx_R)

	def set_tx_R(self,tx_in):
		self.tx_R = tx_in
		self.state_R[4] = tx_in
		self.add_wing_contours()
		self.update_graphs()

	def set_ty_R_spin(self,spin_in):
		self.ty_R_spin = spin_in
		self.ty_R_spin.setMinimum(-10.0)
		self.ty_R_spin.setMaximum(10.0)
		self.ty_R_spin.setDecimals(3)
		self.ty_R_spin.setSingleStep(0.001)
		self.ty_R_spin.setValue(0.0)
		self.set_ty_R(0.0)
		self.ty_R_spin.valueChanged.connect(self.set_ty_R)

	def set_ty_R(self,ty_in):
		self.ty_R = ty_in
		self.state_R[5] = ty_in
		self.add_wing_contours()
		self.update_graphs()

	def set_tz_R_spin(self,spin_in):
		self.tz_R_spin = spin_in
		self.tz_R_spin.setMinimum(-10.0)
		self.tz_R_spin.setMaximum(10.0)
		self.tz_R_spin.setDecimals(3)
		self.tz_R_spin.setSingleStep(0.001)
		self.tz_R_spin.setValue(0.0)
		self.set_tz_R(0.0)
		self.tz_R_spin.valueChanged.connect(self.set_tz_R)

	def set_tz_R(self,tz_in):
		self.tz_R = tz_in
		self.state_R[6] = tz_in
		self.add_wing_contours()
		self.update_graphs()

	def set_beta_R_spin(self,spin_in):
		self.beta_R_spin = spin_in
		self.beta_R_spin.setMinimum(-1.0)
		self.beta_R_spin.setMaximum(1.0)
		self.beta_R_spin.setDecimals(3)
		self.beta_R_spin.setSingleStep(0.001)
		self.beta_R_spin.setValue(0.0)
		self.set_beta_R(0.0)
		self.beta_R_spin.valueChanged.connect(self.set_beta_R)

	def set_beta_R(self,beta_in):
		self.state_R[7] = np.pi*beta_in
		self.add_wing_contours()
		self.update_graphs()

	def reset_state_R(self):
		self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.set_q1_R(0.0)
		self.q1_R_spin.setValue(0.0)
		self.set_q2_R(0.0)
		self.q2_R_spin.setValue(0.0)
		self.set_q3_R(0.0)
		self.q3_R_spin.setValue(0.0)
		self.set_tx_R(0.0)
		self.tx_R_spin.setValue(0.0)
		self.set_ty_R(0.0)
		self.ty_R_spin.setValue(0.0)
		self.set_tz_R(0.0)
		self.tz_R_spin.setValue(0.0)
		self.set_beta_R(0.0)
		self.beta_R_spin.setValue(0.0)

	def project3D_to_uv(self,pts_in):
		uv_pts = []
		for n in range(self.N_cam):
			uv_pts.append(np.dot(self.w2c_matrices[n],pts_in)-self.uv_offset[n]-self.uv_shift[n])
		return uv_pts

	def calculate_key_pts(self):
		key_pts_3D = np.zeros(60)
		key_pts_uv = np.zeros(40*self.N_cam)
		# Left keypoints
		M_list_L = self.state_transform_L()
		pts_0_L = np.ones((4,4))
		pts_0_L[0:3,:] = self.wing_key_pts_L[0:3,[0,1,2,8]]*self.scale_L
		key_0_L = np.dot(M_list_L[0],pts_0_L)
		uv_0_L = self.project3D_to_uv(key_0_L)
		pts_1_L = np.ones((4,3))
		pts_1_L[0:3,:] = self.wing_key_pts_L[0:3,[3,7,9]]*self.scale_L
		key_1_L = np.dot(M_list_L[1],pts_1_L)
		uv_1_L = self.project3D_to_uv(key_1_L)
		pts_2_L = np.ones((4,2))
		pts_2_L[0:3,:] = self.wing_key_pts_L[0:3,[4,6]]*self.scale_L
		key_2_L = np.dot(M_list_L[2],pts_2_L)
		uv_2_L = self.project3D_to_uv(key_2_L)
		pts_3_L = np.ones((4,2))
		pts_3_L[0:3,:] = self.wing_key_pts_L[0:3,[5,4]]*self.scale_L
		key_3_L = np.dot(M_list_L[3],pts_3_L)
		uv_3_L = self.project3D_to_uv(key_3_L)
		# save 3D pts
		key_pts_3D[0:3] 	= pts_0_L[0:3,0]
		key_pts_3D[3:6] 	= pts_0_L[0:3,1]
		key_pts_3D[6:9] 	= pts_0_L[0:3,2]
		key_pts_3D[9:12] 	= pts_1_L[0:3,0]
		key_pts_3D[12:15] 	= pts_2_L[0:3,0]
		key_pts_3D[15:18] 	= pts_3_L[0:3,0]
		key_pts_3D[18:21] 	= pts_2_L[0:3,1]
		key_pts_3D[21:24] 	= pts_1_L[0:3,1]
		key_pts_3D[24:27] 	= pts_0_L[0:3,3]
		key_pts_3D[27:30] 	= pts_1_L[0:3,2]
		# save uv pts
		for n in range(self.N_cam):
			key_pts_uv[(40*n):(40*n+2)] 	= uv_0_L[n][0:2,0]
			key_pts_uv[(40*n+2):(40*n+4)] 	= uv_0_L[n][0:2,1]
			key_pts_uv[(40*n+4):(40*n+6)] 	= uv_0_L[n][0:2,2]
			key_pts_uv[(40*n+6):(40*n+8)] 	= uv_1_L[n][0:2,0]
			key_pts_uv[(40*n+8):(40*n+10)] 	= uv_2_L[n][0:2,0]
			key_pts_uv[(40*n+10):(40*n+12)] = uv_3_L[n][0:2,0]
			key_pts_uv[(40*n+12):(40*n+14)] = uv_2_L[n][0:2,1]
			key_pts_uv[(40*n+14):(40*n+16)] = uv_1_L[n][0:2,1]
			key_pts_uv[(40*n+16):(40*n+18)] = uv_0_L[n][0:2,3]
			key_pts_uv[(40*n+18):(40*n+20)] = uv_1_L[n][0:2,2]
		# Right keypoints
		M_list_R = self.state_transform_R()
		pts_0_R = np.ones((4,4))
		pts_0_R[0:3,:] = self.wing_key_pts_R[0:3,[0,1,2,8]]*self.scale_R
		key_0_R = np.dot(M_list_R[0],pts_0_R)
		uv_0_R = self.project3D_to_uv(key_0_R)
		pts_1_R = np.ones((4,3))
		pts_1_R[0:3,:] = self.wing_key_pts_R[0:3,[3,7,9]]*self.scale_R
		key_1_R = np.dot(M_list_R[1],pts_1_R)
		uv_1_R = self.project3D_to_uv(key_1_R)
		pts_2_R = np.ones((4,2))
		pts_2_R[0:3,:] = self.wing_key_pts_R[0:3,[4,6]]*self.scale_R
		key_2_R = np.dot(M_list_R[2],pts_2_R)
		uv_2_R = self.project3D_to_uv(key_2_R)
		pts_3_R = np.ones((4,2))
		pts_3_R[0:3,:] = self.wing_key_pts_R[0:3,[5,4]]*self.scale_R
		key_3_R = np.dot(M_list_R[3],pts_3_R)
		uv_3_R = self.project3D_to_uv(key_3_R)
		# save 3D pts
		key_pts_3D[30:33] 	= pts_0_R[0:3,0]
		key_pts_3D[33:36] 	= pts_0_R[0:3,1]
		key_pts_3D[36:39] 	= pts_0_R[0:3,2]
		key_pts_3D[39:42] 	= pts_1_R[0:3,0]
		key_pts_3D[42:45] 	= pts_2_R[0:3,0]
		key_pts_3D[45:48] 	= pts_3_R[0:3,0]
		key_pts_3D[48:51] 	= pts_2_R[0:3,1]
		key_pts_3D[51:54] 	= pts_1_R[0:3,1]
		key_pts_3D[54:57] 	= pts_0_R[0:3,3]
		key_pts_3D[57:60] 	= pts_1_R[0:3,2]
		# save uv pts
		for n in range(self.N_cam):
			key_pts_uv[(40*n+20):(40*n+22)] = uv_0_R[n][0:2,0]
			key_pts_uv[(40*n+22):(40*n+24)] = uv_0_R[n][0:2,1]
			key_pts_uv[(40*n+24):(40*n+26)] = uv_0_R[n][0:2,2]
			key_pts_uv[(40*n+26):(40*n+28)] = uv_1_R[n][0:2,0]
			key_pts_uv[(40*n+28):(40*n+30)] = uv_2_R[n][0:2,0]
			key_pts_uv[(40*n+30):(40*n+32)] = uv_3_R[n][0:2,0]
			key_pts_uv[(40*n+32):(40*n+34)] = uv_2_R[n][0:2,1]
			key_pts_uv[(40*n+34):(40*n+36)] = uv_1_R[n][0:2,1]
			key_pts_uv[(40*n+36):(40*n+38)] = uv_0_R[n][0:2,3]
			key_pts_uv[(40*n+38):(40*n+40)] = uv_1_R[n][0:2,2]
		return key_pts_3D, key_pts_uv

	def create_manual_track_dir(self):
		# Check if manual save directory already exists:
		try:
			os.chdir(self.session_folder)
			os.chdir(self.output_folder)
			os.chdir(self.mov_folder)
			print("manual tracking folder exists already")
		except:
			# Create the manual tracking folder
			try:
				os.chdir(self.session_folder)
				os.chdir(self.output_folder)
			except:
				os.chdir(self.session_folder)
				os.mkdir(self.output_folder)
			os.chdir(self.session_folder)
			os.chdir(self.output_folder)
			os.mkdir(self.mov_folder)
			os.chdir(self.mov_folder)
			for n in range(self.N_cam):
				dir_name = 'cam_' + str(n+1)
				os.mkdir(dir_name)
			os.chdir(self.session_folder)
			os.chdir(self.output_folder)
			os.chdir(self.mov_folder)
			os.mkdir('labels')
			print("created manual tracking folder")

	def save_frame(self):
		frame_list = self.load_frame()
		try:
			os.chdir(self.session_folder)
			os.chdir(self.output_folder+'/'+self.mov_folder+'/cam_1')
		except:
			print('Save directory does not exist: creating manual tracking directory')
			self.create_manual_track_dir()
		for i, frame in enumerate(frame_list):
			os.chdir(self.session_folder)
			os.chdir(self.output_folder+'/'+self.mov_folder)
			os.chdir('cam_' + str(i+1))
			img_uint8 = frame*255
			img_uint8 = img_uint8.astype(np.uint8)
			cv2.imwrite('frame_' + str(self.frame_nr) + '.bmp',img_uint8)
			time.sleep(0.001)
		os.chdir(self.session_folder)
		os.chdir(self.output_folder+'/'+self.mov_folder)
		os.chdir('labels')
		state_out = np.zeros((1,18))
		state_out[0,0] = self.scale_L
		state_out[0,1:9] = self.state_L
		state_out[0,9] = self.scale_R
		state_out[0,10:180] = self.state_R
		key_pts_3D, key_pts_uv = self.calculate_key_pts()
		if path.exists('labels.h5'):
			self.hf_label_file = h5py.File('labels.h5', 'r+')
			# Check if frame already exists:
			try:
				state_dat = self.hf_label_file['label_' + str(self.frame_nr)]
				state_dat[...] = state_out
				key_3D_dat = self.hf_label_file['key_pts_3D_' + str(self.frame_nr)]
				key_3D_dat[...] = key_pts_3D
				key_uv_dat = self.hf_label_file['key_pts_uv_' + str(self.frame_nr)]
				key_uv_dat[...] = key_pts_uv
			except:
				self.hf_label_file.create_dataset('label_' + str(self.frame_nr),data=state_out)
				self.hf_label_file.create_dataset('key_pts_3D_' + str(self.frame_nr),data=key_pts_3D)
				self.hf_label_file.create_dataset('key_pts_uv_' + str(self.frame_nr),data=key_pts_uv)
			self.hf_label_file.close()
		else:
			self.hf_label_file = h5py.File('labels.h5', 'w')
			self.hf_label_file.create_dataset('label_' + str(self.frame_nr),data=state_out)
			self.hf_label_file.create_dataset('key_pts_3D_' + str(self.frame_nr),data=key_pts_3D)
			self.hf_label_file.create_dataset('key_pts_uv_' + str(self.frame_nr),data=key_pts_uv)
			self.hf_label_file.close()
		print('saved frame_' + str(self.frame_nr))