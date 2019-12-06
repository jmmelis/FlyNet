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

class ImgViewer1(pg.GraphicsWindow):

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
		self.crop_graph_list = []
		self.crop_cntr_xyz = np.array([[0.0],[0.0],[0.0],[1.0]])
		self.image_centers = [[128.0,128.0],[128.0,128.0],[128.0,128.0]]
		self.crop_window = [[192,192],[192,192],[192,192]]
		self.window_outline = []
		# Image center out
		self.uv_centers = self.image_centers
		self.thorax_center = np.array([[0.0],[0.0],[0.0],[1.0]])

	def set_session_folder(self,session_folder):
		self.session_folder = session_folder

	def set_N_cam(self,N_cam):
		self.N_cam = N_cam

	def set_trigger_mode(self,trigger_mode,trigger_frame):
		self.trigger_mode = trigger_mode
		self.trigger_frame = trigger_frame

	def set_movie_nr(self,mov_nr):
		self.mov_nr = mov_nr-1
		self.mov_folder = self.mov_folders[self.mov_nr]

	def load_bckg_frames(self,bckg_path,bckg_frames,bckg_img_format):
		self.bckg_path = bckg_path
		self.bckg_frames = bckg_frames
		self.bckg_img_format = bckg_img_format
		self.bckg_imgs = []
		for bckg_frame in self.bckg_frames:
			os.chdir(self.bckg_path)
			img_cv = cv2.imread(bckg_frame+self.bckg_img_format,0)
			self.bckg_imgs.append(img_cv/255.0)

	def load_camera_calibration(self,c_params,c2w_matrices,w2c_matrices):
		self.c_params = c_params
		self.c2w_matrices = c2w_matrices
		self.w2c_matrices = w2c_matrices

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
			frame_list.append(img_cv/255.0)
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

	def add_crop_graphs(self):
		self.crop_graph_list = []
		for i in range(self.N_cam):
			self.crop_graph_list.append(Graph(i))
			self.v_list[i].addItem(self.crop_graph_list[i])
			self.crop_txt = ['']
			self.crop_sym = ['o']
			self.crop_clr = ['g']
			crop_uv = np.array([[self.image_centers[i][0],self.image_centers[i][1]]])
			self.crop_graph_list[i].setData(pos=crop_uv, size=2, symbol=self.crop_sym, pxMode=False, text=self.crop_txt, textcolor=self.crop_clr)

	def update_crop_graphs(self):
		for i in range(self.N_cam):
			crop_uv = np.array([[self.image_centers[i][0],self.image_centers[i][1]]])
			self.crop_graph_list[i].setData(pos=crop_uv, size=2, symbol=self.crop_sym, pxMode=False, text=self.crop_txt, textcolor=self.crop_clr)

	def remove_crop_graphs(self):
		if len(self.crop_graph_list)>0:
			for i in range(self.N_cam):
				self.v_list[i].removeItem(self.crop_graph_list[i])
			self.crop_graph_list = []

	def update_3D_points(self,u_prev,v_prev,u_now,v_now,point_nr,cam_nr):
		uv_prev = np.array([[u_prev+(self.c_params[11,cam_nr]-self.c_params[13,cam_nr])/2.0],[self.c_params[10,cam_nr]-(self.c_params[10,cam_nr]-self.c_params[12,cam_nr])/2.0-v_prev],[1.0]])
		xyz_prev = np.squeeze(np.dot(self.c2w_matrices[cam_nr],uv_prev))
		uv_now = np.array([[u_now+(self.c_params[11,cam_nr]-self.c_params[13,cam_nr])/2.0],[self.c_params[10,cam_nr]-(self.c_params[10,cam_nr]-self.c_params[12,cam_nr])/2.0-v_now],[1.0]])
		xyz_now = np.squeeze(np.dot(self.c2w_matrices[cam_nr],uv_now))
		d_xyz = xyz_now-xyz_prev
		# update 3D coordinates:
		self.crop_cntr_xyz[0] = self.crop_cntr_xyz[0]+d_xyz[0]
		self.crop_cntr_xyz[1] = self.crop_cntr_xyz[1]+d_xyz[1]
		self.crop_cntr_xyz[2] = self.crop_cntr_xyz[2]+d_xyz[2]
		# update uv coordinates:
		self.image_centers = []
		for n in range(self.N_cam):
			uv_pts = np.squeeze(np.dot(self.w2c_matrices[n],self.crop_cntr_xyz))
			uv_pts[0] = np.squeeze(uv_pts[0]-(self.c_params[11,n]-self.c_params[13,n])/2.0)
			uv_pts[1] = np.squeeze(self.c_params[10,n]-uv_pts[1]-(self.c_params[10,n]-self.c_params[12,n])/2.0)
			self.image_centers.append([uv_pts[0],uv_pts[1]])
		self.add_crop_window()

	def setMouseCallbacks(self):
		def onMouseDragCallback(data):
			cam_nr = int(data[3])
			point_nr = int(data[2])
			if point_nr<1:
				u_prev = self.image_centers[cam_nr][0]
				v_prev = self.image_centers[cam_nr][1]
				u_now = data[0]
				v_now = data[1]
			self.update_3D_points(u_prev,v_prev,u_now,v_now,point_nr,cam_nr)
			self.update_crop_graphs()
			self.update_table_widget()

		for i in range(self.N_cam):
			self.crop_graph_list[i].setOnMouseDragCallback(onMouseDragCallback)

	def add_crop_window(self):
		self.remove_crop_window()
		# Add crop window:
		self.window_outline = []
		for n in range(self.N_cam):
			uv_window = np.zeros((4,2))
			uv_window[0,0] = np.round(self.image_centers[n][0]-self.crop_window[n][0]/2.0)
			uv_window[0,1] = np.round(self.image_centers[n][1]-self.crop_window[n][1]/2.0)
			uv_window[1,0] = np.round(self.image_centers[n][0]-self.crop_window[n][0]/2.0)
			uv_window[1,1] = np.round(self.image_centers[n][1]+self.crop_window[n][1]/2.0)
			uv_window[2,0] = np.round(self.image_centers[n][0]+self.crop_window[n][0]/2.0)
			uv_window[2,1] = np.round(self.image_centers[n][1]+self.crop_window[n][1]/2.0)
			uv_window[3,0] = np.round(self.image_centers[n][0]+self.crop_window[n][0]/2.0)
			uv_window[3,1] = np.round(self.image_centers[n][1]-self.crop_window[n][1]/2.0)
			line_A = pg.PlotCurveItem(x=uv_window[[0,1],0],y=uv_window[[0,1],1],pen=[0,255,0])
			line_B = pg.PlotCurveItem(x=uv_window[[1,2],0],y=uv_window[[1,2],1],pen=[0,255,0])
			line_C = pg.PlotCurveItem(x=uv_window[[2,3],0],y=uv_window[[2,3],1],pen=[0,255,0])
			line_D = pg.PlotCurveItem(x=uv_window[[3,0],0],y=uv_window[[3,0],1],pen=[0,255,0])
			self.window_outline.append([line_A,line_B,line_C,line_D])
			self.v_list[n].addItem(line_A)
			self.v_list[n].addItem(line_B)
			self.v_list[n].addItem(line_C)
			self.v_list[n].addItem(line_D)

	def remove_crop_window(self):
		if len(self.window_outline) == self.N_cam:
			for n in range(self.N_cam):
				for i, outline in enumerate(self.window_outline[n]):
					self.v_list[n].removeItem(outline)
		self.window_outline = []

	def set_img_cntr(self):
		self.uv_centers = self.image_centers
		print('image centers:')
		print(self.uv_centers)
		self.thorax_center = self.crop_cntr_xyz
		print('thorax center: ')
		print(self.thorax_center)

	def set_table_widget(self,table_in):
		self.uv_table = table_in
		self.uv_table.setRowCount(self.N_cam+1)
		self.uv_table.setColumnCount(3)
		self.uv_table.setItem(0,0,QTableWidgetItem('---'))
		self.uv_table.setItem(0,1,QTableWidgetItem('u'))
		self.uv_table.setItem(0,2,QTableWidgetItem('v'))
		for n in range(self.N_cam):
			self.uv_table.setItem(1+n,0,QTableWidgetItem('Camera ' + str(n+1)))
		self.uv_table.resizeColumnsToContents()

	def update_table_widget(self):
		for n in range(self.N_cam):
			self.uv_table.setItem(1+n,1,QTableWidgetItem(str(np.around(self.image_centers[n][0],decimals=2))))
			self.uv_table.setItem(1+n,2,QTableWidgetItem(str(np.around(self.image_centers[n][1],decimals=2))))
		self.uv_table.resizeColumnsToContents()