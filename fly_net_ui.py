# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fly_net.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1810, 911)
        MainWindow.setMinimumSize(QtCore.QSize(1810, 911))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.DataLoadTab = QtWidgets.QWidget()
        self.DataLoadTab.setObjectName("DataLoadTab")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.DataLoadTab)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.frame = QtWidgets.QFrame(self.DataLoadTab)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 4)
        spacerItem = QtWidgets.QSpacerItem(116, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 6, 1, 2)
        self.line = QtWidgets.QFrame(self.frame_2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 0, 1, 8)
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(168, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 2, 1, 6)
        self.seq_folder_disp = QtWidgets.QLineEdit(self.frame_2)
        self.seq_folder_disp.setObjectName("seq_folder_disp")
        self.gridLayout.addWidget(self.seq_folder_disp, 3, 0, 1, 3)
        self.select_seq_btn = QtWidgets.QPushButton(self.frame_2)
        self.select_seq_btn.setObjectName("select_seq_btn")
        self.gridLayout.addWidget(self.select_seq_btn, 3, 7, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(151, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 4, 3, 1, 5)
        self.bckg_folder_disp = QtWidgets.QLineEdit(self.frame_2)
        self.bckg_folder_disp.setObjectName("bckg_folder_disp")
        self.gridLayout.addWidget(self.bckg_folder_disp, 5, 0, 1, 3)
        self.select_bckg_btn = QtWidgets.QPushButton(self.frame_2)
        self.select_bckg_btn.setObjectName("select_bckg_btn")
        self.gridLayout.addWidget(self.select_bckg_btn, 5, 7, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 6, 0, 1, 2)
        spacerItem3 = QtWidgets.QSpacerItem(151, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 6, 3, 1, 5)
        self.calib_folder_disp = QtWidgets.QLineEdit(self.frame_2)
        self.calib_folder_disp.setObjectName("calib_folder_disp")
        self.gridLayout.addWidget(self.calib_folder_disp, 7, 0, 1, 3)
        self.select_calib_btn = QtWidgets.QPushButton(self.frame_2)
        self.select_calib_btn.setObjectName("select_calib_btn")
        self.gridLayout.addWidget(self.select_calib_btn, 7, 7, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.frame_2)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 8, 0, 1, 8)
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 9, 0, 1, 3)
        spacerItem4 = QtWidgets.QSpacerItem(133, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 9, 4, 1, 4)
        self.movie_table = QtWidgets.QTableWidget(self.frame_2)
        self.movie_table.setObjectName("movie_table")
        self.movie_table.setColumnCount(0)
        self.movie_table.setRowCount(0)
        self.gridLayout.addWidget(self.movie_table, 10, 0, 1, 8)
        self.add_movie_btn = QtWidgets.QPushButton(self.frame_2)
        self.add_movie_btn.setObjectName("add_movie_btn")
        self.gridLayout.addWidget(self.add_movie_btn, 11, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(83, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem5, 11, 1, 1, 5)
        self.reset_movie_btn = QtWidgets.QPushButton(self.frame_2)
        self.reset_movie_btn.setObjectName("reset_movie_btn")
        self.gridLayout.addWidget(self.reset_movie_btn, 11, 7, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.frame_2)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 12, 0, 1, 8)
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 13, 0, 1, 1)
        self.trigger_mode_combo = QtWidgets.QComboBox(self.frame_2)
        self.trigger_mode_combo.setObjectName("trigger_mode_combo")
        self.gridLayout.addWidget(self.trigger_mode_combo, 13, 1, 1, 6)
        spacerItem6 = QtWidgets.QSpacerItem(106, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem6, 13, 7, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.frame_2)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 14, 0, 1, 8)
        spacerItem7 = QtWidgets.QSpacerItem(288, 297, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem7, 15, 0, 1, 8)
        self.line_5 = QtWidgets.QFrame(self.frame_2)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout.addWidget(self.line_5, 16, 0, 1, 8)
        spacerItem8 = QtWidgets.QSpacerItem(173, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem8, 17, 0, 1, 5)
        self.load_seq_btn = QtWidgets.QPushButton(self.frame_2)
        self.load_seq_btn.setObjectName("load_seq_btn")
        self.gridLayout.addWidget(self.load_seq_btn, 17, 5, 1, 3)
        self.horizontalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.img_viewer_1 = ImgViewer1(self.frame_3)
        self.img_viewer_1.setMinimumSize(QtCore.QSize(1404, 561))
        self.img_viewer_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.img_viewer_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.img_viewer_1.setObjectName("img_viewer_1")
        self.verticalLayout_3.addWidget(self.img_viewer_1)
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setMinimumSize(QtCore.QSize(1404, 211))
        self.frame_5.setMaximumSize(QtCore.QSize(16777215, 211))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setMinimumSize(QtCore.QSize(91, 0))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_7 = QtWidgets.QLabel(self.frame_6)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.movie_spin_1 = QtWidgets.QSpinBox(self.frame_6)
        self.movie_spin_1.setObjectName("movie_spin_1")
        self.verticalLayout.addWidget(self.movie_spin_1)
        self.label_8 = QtWidgets.QLabel(self.frame_6)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.frame_spin_1 = QtWidgets.QSpinBox(self.frame_6)
        self.frame_spin_1.setObjectName("frame_spin_1")
        self.verticalLayout.addWidget(self.frame_spin_1)
        spacerItem9 = QtWidgets.QSpacerItem(68, 58, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem9)
        self.horizontalLayout_2.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_5)
        self.frame_7.setMinimumSize(QtCore.QSize(191, 191))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_7)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_9 = QtWidgets.QLabel(self.frame_7)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 0, 1, 2)
        self.line_6 = QtWidgets.QFrame(self.frame_7)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout_2.addWidget(self.line_6, 1, 0, 1, 2)
        self.seg_view_btn = QtWidgets.QRadioButton(self.frame_7)
        self.seg_view_btn.setObjectName("seg_view_btn")
        self.gridLayout_2.addWidget(self.seg_view_btn, 2, 0, 1, 2)
        self.img_view_btn = QtWidgets.QRadioButton(self.frame_7)
        self.img_view_btn.setObjectName("img_view_btn")
        self.gridLayout_2.addWidget(self.img_view_btn, 3, 0, 1, 2)
        self.line_7 = QtWidgets.QFrame(self.frame_7)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout_2.addWidget(self.line_7, 4, 0, 1, 2)
        self.label_10 = QtWidgets.QLabel(self.frame_7)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 5, 0, 1, 1)
        self.camera_view_spin_1 = QtWidgets.QSpinBox(self.frame_7)
        self.camera_view_spin_1.setObjectName("camera_view_spin_1")
        self.gridLayout_2.addWidget(self.camera_view_spin_1, 5, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame_7)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 6, 0, 1, 1)
        self.mask_spin = QtWidgets.QSpinBox(self.frame_7)
        self.mask_spin.setObjectName("mask_spin")
        self.gridLayout_2.addWidget(self.mask_spin, 6, 1, 1, 1)
        self.add_mask_btn = QtWidgets.QPushButton(self.frame_7)
        self.add_mask_btn.setObjectName("add_mask_btn")
        self.gridLayout_2.addWidget(self.add_mask_btn, 7, 0, 1, 1)
        self.reset_masks_btn = QtWidgets.QPushButton(self.frame_7)
        self.reset_masks_btn.setObjectName("reset_masks_btn")
        self.gridLayout_2.addWidget(self.reset_masks_btn, 7, 1, 1, 1)
        self.horizontalLayout_2.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.frame_5)
        self.frame_8.setMinimumSize(QtCore.QSize(411, 191))
        self.frame_8.setMaximumSize(QtCore.QSize(16777215, 191))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_8)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_12 = QtWidgets.QLabel(self.frame_8)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 0, 0, 1, 2)
        spacerItem10 = QtWidgets.QSpacerItem(312, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem10, 2, 0, 1, 3)
        self.set_img_cntr_btn = QtWidgets.QPushButton(self.frame_8)
        self.set_img_cntr_btn.setObjectName("set_img_cntr_btn")
        self.gridLayout_3.addWidget(self.set_img_cntr_btn, 2, 3, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(270, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem11, 0, 2, 1, 2)
        self.img_center_table = QtWidgets.QTableWidget(self.frame_8)
        self.img_center_table.setObjectName("img_center_table")
        self.img_center_table.setColumnCount(0)
        self.img_center_table.setRowCount(0)
        self.gridLayout_3.addWidget(self.img_center_table, 1, 0, 1, 4)
        self.horizontalLayout_2.addWidget(self.frame_8)
        spacerItem12 = QtWidgets.QSpacerItem(458, 178, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem12)
        self.frame_9 = QtWidgets.QFrame(self.frame_5)
        self.frame_9.setMinimumSize(QtCore.QSize(161, 191))
        self.frame_9.setMaximumSize(QtCore.QSize(16777215, 191))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem13 = QtWidgets.QSpacerItem(128, 137, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem13)
        self.continue_1_btn = QtWidgets.QPushButton(self.frame_9)
        self.continue_1_btn.setObjectName("continue_1_btn")
        self.verticalLayout_2.addWidget(self.continue_1_btn)
        self.horizontalLayout_2.addWidget(self.frame_9)
        self.verticalLayout_3.addWidget(self.frame_5)
        self.horizontalLayout.addWidget(self.frame_3)
        self.verticalLayout_7.addWidget(self.frame)
        self.tabWidget.addTab(self.DataLoadTab, "")
        self.AnnotationTab = QtWidgets.QWidget()
        self.AnnotationTab.setObjectName("AnnotationTab")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.AnnotationTab)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_4 = QtWidgets.QFrame(self.AnnotationTab)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.img_viewer_2 = ImgViewer2(self.frame_4)
        self.img_viewer_2.setMinimumSize(QtCore.QSize(1736, 601))
        self.img_viewer_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.img_viewer_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.img_viewer_2.setObjectName("img_viewer_2")
        self.verticalLayout_4.addWidget(self.img_viewer_2)
        self.frame_11 = QtWidgets.QFrame(self.frame_4)
        self.frame_11.setMinimumSize(QtCore.QSize(1736, 198))
        self.frame_11.setMaximumSize(QtCore.QSize(16777215, 198))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_10 = QtWidgets.QFrame(self.frame_11)
        self.frame_10.setMinimumSize(QtCore.QSize(91, 0))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_16 = QtWidgets.QLabel(self.frame_10)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_5.addWidget(self.label_16)
        self.movie_spin_2 = QtWidgets.QSpinBox(self.frame_10)
        self.movie_spin_2.setObjectName("movie_spin_2")
        self.verticalLayout_5.addWidget(self.movie_spin_2)
        self.label_17 = QtWidgets.QLabel(self.frame_10)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_5.addWidget(self.label_17)
        self.frame_spin_2 = QtWidgets.QSpinBox(self.frame_10)
        self.frame_spin_2.setObjectName("frame_spin_2")
        self.verticalLayout_5.addWidget(self.frame_spin_2)
        spacerItem14 = QtWidgets.QSpacerItem(58, 45, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem14)
        self.horizontalLayout_3.addWidget(self.frame_10)
        self.frame_12 = QtWidgets.QFrame(self.frame_11)
        self.frame_12.setMinimumSize(QtCore.QSize(281, 171))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_12)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_18 = QtWidgets.QLabel(self.frame_12)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 0, 0, 1, 3)
        self.line_9 = QtWidgets.QFrame(self.frame_12)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.gridLayout_4.addWidget(self.line_9, 1, 0, 1, 3)
        self.label_19 = QtWidgets.QLabel(self.frame_12)
        self.label_19.setObjectName("label_19")
        self.gridLayout_4.addWidget(self.label_19, 2, 0, 1, 1)
        self.network_combo = QtWidgets.QComboBox(self.frame_12)
        self.network_combo.setObjectName("network_combo")
        self.gridLayout_4.addWidget(self.network_combo, 2, 1, 1, 2)
        self.weight_file_disp = QtWidgets.QLineEdit(self.frame_12)
        self.weight_file_disp.setObjectName("weight_file_disp")
        self.gridLayout_4.addWidget(self.weight_file_disp, 3, 0, 1, 2)
        self.select_weights_btn = QtWidgets.QPushButton(self.frame_12)
        self.select_weights_btn.setObjectName("select_weights_btn")
        self.gridLayout_4.addWidget(self.select_weights_btn, 3, 2, 1, 1)
        spacerItem15 = QtWidgets.QSpacerItem(157, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem15, 4, 0, 1, 2)
        self.predict_state_btn = QtWidgets.QPushButton(self.frame_12)
        self.predict_state_btn.setObjectName("predict_state_btn")
        self.gridLayout_4.addWidget(self.predict_state_btn, 4, 2, 1, 1)
        self.horizontalLayout_3.addWidget(self.frame_12)
        self.frame_16 = QtWidgets.QFrame(self.frame_11)
        self.frame_16.setMinimumSize(QtCore.QSize(141, 171))
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_16)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_13 = QtWidgets.QLabel(self.frame_16)
        self.label_13.setObjectName("label_13")
        self.gridLayout_7.addWidget(self.label_13, 0, 0, 1, 2)
        self.line_8 = QtWidgets.QFrame(self.frame_16)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.gridLayout_7.addWidget(self.line_8, 1, 0, 1, 2)
        self.label_14 = QtWidgets.QLabel(self.frame_16)
        self.label_14.setObjectName("label_14")
        self.gridLayout_7.addWidget(self.label_14, 2, 0, 1, 1)
        self.camera_view_spin_2 = QtWidgets.QSpinBox(self.frame_16)
        self.camera_view_spin_2.setObjectName("camera_view_spin_2")
        self.gridLayout_7.addWidget(self.camera_view_spin_2, 2, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.frame_16)
        self.label_15.setObjectName("label_15")
        self.gridLayout_7.addWidget(self.label_15, 3, 0, 1, 1)
        self.u_shift_spin = QtWidgets.QSpinBox(self.frame_16)
        self.u_shift_spin.setObjectName("u_shift_spin")
        self.gridLayout_7.addWidget(self.u_shift_spin, 3, 1, 1, 1)
        self.label_41 = QtWidgets.QLabel(self.frame_16)
        self.label_41.setObjectName("label_41")
        self.gridLayout_7.addWidget(self.label_41, 4, 0, 1, 1)
        self.v_shift_spin = QtWidgets.QSpinBox(self.frame_16)
        self.v_shift_spin.setObjectName("v_shift_spin")
        self.gridLayout_7.addWidget(self.v_shift_spin, 4, 1, 1, 1)
        spacerItem16 = QtWidgets.QSpacerItem(118, 27, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem16, 5, 0, 1, 2)
        self.horizontalLayout_3.addWidget(self.frame_16)
        self.frame_13 = QtWidgets.QFrame(self.frame_11)
        self.frame_13.setMinimumSize(QtCore.QSize(381, 171))
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_13)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_20 = QtWidgets.QLabel(self.frame_13)
        self.label_20.setObjectName("label_20")
        self.gridLayout_6.addWidget(self.label_20, 0, 0, 1, 2)
        spacerItem17 = QtWidgets.QSpacerItem(163, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem17, 0, 2, 1, 3)
        self.label_21 = QtWidgets.QLabel(self.frame_13)
        self.label_21.setObjectName("label_21")
        self.gridLayout_6.addWidget(self.label_21, 0, 5, 1, 1)
        self.scale_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.scale_L_spin.setObjectName("scale_L_spin")
        self.gridLayout_6.addWidget(self.scale_L_spin, 0, 6, 1, 1)
        self.line_10 = QtWidgets.QFrame(self.frame_13)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.gridLayout_6.addWidget(self.line_10, 1, 0, 1, 7)
        self.label_22 = QtWidgets.QLabel(self.frame_13)
        self.label_22.setObjectName("label_22")
        self.gridLayout_6.addWidget(self.label_22, 2, 0, 2, 1)
        self.q1_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.q1_L_spin.setObjectName("q1_L_spin")
        self.gridLayout_6.addWidget(self.q1_L_spin, 2, 1, 2, 2)
        self.label_25 = QtWidgets.QLabel(self.frame_13)
        self.label_25.setObjectName("label_25")
        self.gridLayout_6.addWidget(self.label_25, 2, 3, 2, 1)
        self.tx_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.tx_L_spin.setObjectName("tx_L_spin")
        self.gridLayout_6.addWidget(self.tx_L_spin, 2, 4, 2, 1)
        self.beta_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.beta_L_spin.setObjectName("beta_L_spin")
        self.gridLayout_6.addWidget(self.beta_L_spin, 2, 6, 2, 1)
        self.label_29 = QtWidgets.QLabel(self.frame_13)
        self.label_29.setObjectName("label_29")
        self.gridLayout_6.addWidget(self.label_29, 3, 5, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.frame_13)
        self.label_23.setObjectName("label_23")
        self.gridLayout_6.addWidget(self.label_23, 4, 0, 1, 1)
        self.q2_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.q2_L_spin.setObjectName("q2_L_spin")
        self.gridLayout_6.addWidget(self.q2_L_spin, 4, 1, 1, 2)
        self.label_26 = QtWidgets.QLabel(self.frame_13)
        self.label_26.setObjectName("label_26")
        self.gridLayout_6.addWidget(self.label_26, 4, 3, 1, 1)
        self.ty_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.ty_L_spin.setObjectName("ty_L_spin")
        self.gridLayout_6.addWidget(self.ty_L_spin, 4, 4, 1, 1)
        self.reset_L_btn = QtWidgets.QPushButton(self.frame_13)
        self.reset_L_btn.setObjectName("reset_L_btn")
        self.gridLayout_6.addWidget(self.reset_L_btn, 4, 5, 1, 2)
        self.label_24 = QtWidgets.QLabel(self.frame_13)
        self.label_24.setObjectName("label_24")
        self.gridLayout_6.addWidget(self.label_24, 5, 0, 1, 1)
        self.q3_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.q3_L_spin.setObjectName("q3_L_spin")
        self.gridLayout_6.addWidget(self.q3_L_spin, 5, 1, 1, 2)
        self.label_27 = QtWidgets.QLabel(self.frame_13)
        self.label_27.setObjectName("label_27")
        self.gridLayout_6.addWidget(self.label_27, 5, 3, 1, 1)
        self.tz_L_spin = QtWidgets.QDoubleSpinBox(self.frame_13)
        self.tz_L_spin.setObjectName("tz_L_spin")
        self.gridLayout_6.addWidget(self.tz_L_spin, 5, 4, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(358, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem18, 6, 0, 1, 7)
        self.horizontalLayout_3.addWidget(self.frame_13)
        self.frame_14 = QtWidgets.QFrame(self.frame_11)
        self.frame_14.setMinimumSize(QtCore.QSize(381, 171))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_14)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_30 = QtWidgets.QLabel(self.frame_14)
        self.label_30.setObjectName("label_30")
        self.gridLayout_5.addWidget(self.label_30, 0, 0, 1, 2)
        spacerItem19 = QtWidgets.QSpacerItem(155, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem19, 0, 2, 1, 3)
        self.label_31 = QtWidgets.QLabel(self.frame_14)
        self.label_31.setObjectName("label_31")
        self.gridLayout_5.addWidget(self.label_31, 0, 5, 1, 1)
        self.scale_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.scale_R_spin.setObjectName("scale_R_spin")
        self.gridLayout_5.addWidget(self.scale_R_spin, 0, 6, 1, 1)
        self.line_11 = QtWidgets.QFrame(self.frame_14)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_5.addWidget(self.line_11, 1, 0, 1, 7)
        self.label_32 = QtWidgets.QLabel(self.frame_14)
        self.label_32.setObjectName("label_32")
        self.gridLayout_5.addWidget(self.label_32, 2, 0, 2, 1)
        self.q1_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.q1_R_spin.setObjectName("q1_R_spin")
        self.gridLayout_5.addWidget(self.q1_R_spin, 2, 1, 2, 2)
        self.label_35 = QtWidgets.QLabel(self.frame_14)
        self.label_35.setObjectName("label_35")
        self.gridLayout_5.addWidget(self.label_35, 2, 3, 2, 1)
        self.tx_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.tx_R_spin.setObjectName("tx_R_spin")
        self.gridLayout_5.addWidget(self.tx_R_spin, 2, 4, 2, 1)
        self.beta_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.beta_R_spin.setObjectName("beta_R_spin")
        self.gridLayout_5.addWidget(self.beta_R_spin, 2, 6, 2, 1)
        self.label_39 = QtWidgets.QLabel(self.frame_14)
        self.label_39.setObjectName("label_39")
        self.gridLayout_5.addWidget(self.label_39, 3, 5, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.frame_14)
        self.label_33.setObjectName("label_33")
        self.gridLayout_5.addWidget(self.label_33, 4, 0, 1, 1)
        self.q2_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.q2_R_spin.setObjectName("q2_R_spin")
        self.gridLayout_5.addWidget(self.q2_R_spin, 4, 1, 1, 2)
        self.label_36 = QtWidgets.QLabel(self.frame_14)
        self.label_36.setObjectName("label_36")
        self.gridLayout_5.addWidget(self.label_36, 4, 3, 1, 1)
        self.ty_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.ty_R_spin.setObjectName("ty_R_spin")
        self.gridLayout_5.addWidget(self.ty_R_spin, 4, 4, 1, 1)
        self.reset_R_btn = QtWidgets.QPushButton(self.frame_14)
        self.reset_R_btn.setObjectName("reset_R_btn")
        self.gridLayout_5.addWidget(self.reset_R_btn, 4, 5, 1, 2)
        self.label_34 = QtWidgets.QLabel(self.frame_14)
        self.label_34.setObjectName("label_34")
        self.gridLayout_5.addWidget(self.label_34, 5, 0, 1, 1)
        self.q3_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.q3_R_spin.setObjectName("q3_R_spin")
        self.gridLayout_5.addWidget(self.q3_R_spin, 5, 1, 1, 2)
        self.label_37 = QtWidgets.QLabel(self.frame_14)
        self.label_37.setObjectName("label_37")
        self.gridLayout_5.addWidget(self.label_37, 5, 3, 1, 1)
        self.tz_R_spin = QtWidgets.QDoubleSpinBox(self.frame_14)
        self.tz_R_spin.setObjectName("tz_R_spin")
        self.gridLayout_5.addWidget(self.tz_R_spin, 5, 4, 1, 1)
        spacerItem20 = QtWidgets.QSpacerItem(358, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem20, 6, 0, 1, 7)
        self.horizontalLayout_3.addWidget(self.frame_14)
        spacerItem21 = QtWidgets.QSpacerItem(358, 168, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem21)
        self.frame_15 = QtWidgets.QFrame(self.frame_11)
        self.frame_15.setMinimumSize(QtCore.QSize(191, 171))
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_15)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_40 = QtWidgets.QLabel(self.frame_15)
        self.label_40.setObjectName("label_40")
        self.verticalLayout_6.addWidget(self.label_40)
        self.line_12 = QtWidgets.QFrame(self.frame_15)
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.verticalLayout_6.addWidget(self.line_12)
        self.create_label_dir_btn = QtWidgets.QPushButton(self.frame_15)
        self.create_label_dir_btn.setObjectName("create_label_dir_btn")
        self.verticalLayout_6.addWidget(self.create_label_dir_btn)
        spacerItem22 = QtWidgets.QSpacerItem(158, 61, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem22)
        self.save_label_btn = QtWidgets.QPushButton(self.frame_15)
        self.save_label_btn.setObjectName("save_label_btn")
        self.verticalLayout_6.addWidget(self.save_label_btn)
        self.horizontalLayout_3.addWidget(self.frame_15)
        self.verticalLayout_4.addWidget(self.frame_11)
        self.horizontalLayout_5.addWidget(self.frame_4)
        self.tabWidget.addTab(self.AnnotationTab, "")
        self.horizontalLayout_4.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Select fligth sequence:</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Sequence folder:"))
        self.select_seq_btn.setText(_translate("MainWindow", "select"))
        self.label_3.setText(_translate("MainWindow", "Background folder:"))
        self.select_bckg_btn.setText(_translate("MainWindow", "select"))
        self.label_4.setText(_translate("MainWindow", "Calibration folder:"))
        self.select_calib_btn.setText(_translate("MainWindow", "select"))
        self.label_5.setText(_translate("MainWindow", "Select movie folders:"))
        self.add_movie_btn.setText(_translate("MainWindow", "add movie"))
        self.reset_movie_btn.setText(_translate("MainWindow", "reset"))
        self.label_6.setText(_translate("MainWindow", "trigger mode:"))
        self.load_seq_btn.setText(_translate("MainWindow", "load data"))
        self.label_7.setText(_translate("MainWindow", "Movie nr:"))
        self.label_8.setText(_translate("MainWindow", "Frame nr:"))
        self.label_9.setText(_translate("MainWindow", "Select image masks:"))
        self.seg_view_btn.setText(_translate("MainWindow", "Segmentation view"))
        self.img_view_btn.setText(_translate("MainWindow", "Image view"))
        self.label_10.setText(_translate("MainWindow", "Camera view:"))
        self.label_11.setText(_translate("MainWindow", "Select mask:"))
        self.add_mask_btn.setText(_translate("MainWindow", "add mask"))
        self.reset_masks_btn.setText(_translate("MainWindow", "reset"))
        self.label_12.setText(_translate("MainWindow", "Select image center:"))
        self.set_img_cntr_btn.setText(_translate("MainWindow", "set image center"))
        self.continue_1_btn.setText(_translate("MainWindow", "continue"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.DataLoadTab), _translate("MainWindow", "Load data"))
        self.label_16.setText(_translate("MainWindow", "Movie nr:"))
        self.label_17.setText(_translate("MainWindow", "Frame nr:"))
        self.label_18.setText(_translate("MainWindow", "Load predictor network:"))
        self.label_19.setText(_translate("MainWindow", "Select network:"))
        self.select_weights_btn.setText(_translate("MainWindow", "load weights"))
        self.predict_state_btn.setText(_translate("MainWindow", "predict"))
        self.label_13.setText(_translate("MainWindow", "Set uv-shift:"))
        self.label_14.setText(_translate("MainWindow", "Camera:"))
        self.label_15.setText(_translate("MainWindow", "u-shift:"))
        self.label_41.setText(_translate("MainWindow", "v-shift:"))
        self.label_20.setText(_translate("MainWindow", "Left wing:"))
        self.label_21.setText(_translate("MainWindow", "Scale:"))
        self.label_22.setText(_translate("MainWindow", "q1:"))
        self.label_25.setText(_translate("MainWindow", "tx:"))
        self.label_29.setText(_translate("MainWindow", "bending:"))
        self.label_23.setText(_translate("MainWindow", "q2:"))
        self.label_26.setText(_translate("MainWindow", "ty:"))
        self.reset_L_btn.setText(_translate("MainWindow", "reset state"))
        self.label_24.setText(_translate("MainWindow", "q3:"))
        self.label_27.setText(_translate("MainWindow", "tz:"))
        self.label_30.setText(_translate("MainWindow", "Right wing:"))
        self.label_31.setText(_translate("MainWindow", "Scale:"))
        self.label_32.setText(_translate("MainWindow", "q1:"))
        self.label_35.setText(_translate("MainWindow", "tx:"))
        self.label_39.setText(_translate("MainWindow", "bending:"))
        self.label_33.setText(_translate("MainWindow", "q2:"))
        self.label_36.setText(_translate("MainWindow", "ty:"))
        self.reset_R_btn.setText(_translate("MainWindow", "reset state"))
        self.label_34.setText(_translate("MainWindow", "q3:"))
        self.label_37.setText(_translate("MainWindow", "tz:"))
        self.label_40.setText(_translate("MainWindow", "Save annotation:"))
        self.create_label_dir_btn.setText(_translate("MainWindow", "create save directory"))
        self.save_label_btn.setText(_translate("MainWindow", "save current label"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.AnnotationTab), _translate("MainWindow", "Annotate data"))


from ImgViewer1 import ImgViewer1
from ImgViewer2 import ImgViewer2
