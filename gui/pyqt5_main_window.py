# -*- coding: utf-8 -*-
"""
PyQt5主窗口GUI
PyQt5 Main Window GUI for Smart Product Recognition System
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import json
import logging
from datetime import datetime
import threading
from typing import Dict, List, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QLabel, QPushButton, QComboBox, QProgressBar, QFileDialog,
    QMessageBox, QStatusBar, QFrame, QScrollArea, QTableWidget, 
    QTableWidgetItem, QHeaderView, QGroupBox, QSizePolicy, QButtonGroup, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QTextCodec
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor, QImage, QFontDatabase

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GUI_CONFIG, OUTPUT_CONFIG
from core.simple_detection_engine import SimpleDetectionEngine
from core.simple_information_extractor import SimpleInformationExtractor
from core.image_processor import ImageProcessor
from core.gemini_health_analyzer import GeminiHealthAnalyzer
from utils.text_output import TextOutputManager

class DetectionWorker(QThread):
    """检测工作线程"""
    finished = pyqtSignal(dict, dict, object)  # detection_results, extraction_results, processed_image
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, detection_engine, info_extractor, image, scenario):
        super().__init__()
        self.detection_engine = detection_engine
        self.info_extractor = info_extractor
        self.image = image
        self.scenario = scenario
    
    def run(self):
        try:
            self.progress.emit("正在进行商品检测...")
            
            # 执行检测
            results = self.detection_engine.comprehensive_detection(self.image)
            
            self.progress.emit("正在绘制检测结果...")
            
            # 绘制检测结果
            all_detections = []
            for category in ['products', 'regions', 'texts']:
                all_detections.extend(results.get(category, []))
            
            processed_img = self.detection_engine.draw_detections(
                self.image.copy(),
                all_detections
            )
            
            self.progress.emit("正在提取信息...")
            
            # 提取信息
            extraction_results = self.info_extractor.extract_comprehensive_info(
                self.image,
                results
            )
            
            self.finished.emit(results, extraction_results, processed_img)
            
        except Exception as e:
            self.error.emit(str(e))

class PyQt5MainWindow(QMainWindow):
    """PyQt5主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setup_font_and_encoding()  # 设置字体和编码
        self.setup_logging()
        self.init_variables()
        self.init_ui()
        self.init_components()
        
    def setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(__name__)
        
        # 创建日志处理器
        log_file = Path(OUTPUT_CONFIG['log_dir']) / f"pyqt5_gui_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def setup_font_and_encoding(self):
        """设置字体和编码以支持中文显示"""
        # 设置文本编码
        QTextCodec.setCodecForLocale(QTextCodec.codecForName("UTF-8"))
        
        # 加载中文字体
        font_db = QFontDatabase()
        
        # 尝试加载系统中文字体
        chinese_fonts = [
            "Microsoft YaHei UI",
            "Microsoft YaHei", 
            "SimHei",
            "SimSun",
            "KaiTi",
            "FangSong"
        ]
        
        self.default_font = None
        for font_name in chinese_fonts:
            if font_db.families().count(font_name) > 0:
                self.default_font = QFont(font_name, 12)
                break        
        if self.default_font is None:
            # 如果没有找到中文字体，使用默认字体
            self.default_font = QFont("Arial Unicode MS", 12)
        
        # 设置应用程序默认字体
        QApplication.instance().setFont(self.default_font)
    
    def init_variables(self):
        """初始化变量"""
        self.current_image = None
        self.current_image_path = None
        self.detection_results = None
        self.extraction_results = None
        self.processed_image = None
        self.detection_engine = None
        self.gemini_analyzer = None  # Gemini健康分析器
        self.health_analysis_results = None  # 健康分析结果
        self.zoom_factor = 1.0
        self.current_scenario = "personal_shopping"  # 默认场景
        
    def setup_window_icon(self):
        """设置窗口图标"""
        try:
            # 尝试加载应用图标
            icon_path = Path(__file__).parent.parent / "assets" / "icons" / "app_icon_32x32.png"
            if icon_path.exists():
                icon = QIcon(str(icon_path))
                self.setWindowIcon(icon)
                # 同时设置应用程序图标（用于任务栏）
                QApplication.instance().setWindowIcon(icon)
                self.logger.info(f"已设置窗口图标: {icon_path}")
            else:
                self.logger.warning(f"图标文件不存在: {icon_path}")
        except Exception as e:
            self.logger.error(f"设置窗口图标失败: {e}")
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(GUI_CONFIG['window_title'])
        
        # 设置应用图标
        self.setup_window_icon()
        
        # 设置合理的窗口尺寸，支持更好的伸缩性
        self.setGeometry(100, 100, 2000, 1300)
        self.setMinimumSize(2600, 1400)  # 设置合理的最小尺寸，支持小屏幕
        
        # 设置应用样式 - 优化字体大小和布局，提高伸缩性
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
                font-size: 24px;
                font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', Arial, sans-serif;
                font-weight: 500;
            }
            QTabWidget::pane {
                border: 3px solid #c0c0c0;
                background-color: white;
                font-size: 24px;
                margin: 0px;
                padding: 5px;
            }
            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #e8f4fd, stop: 1 #d1e7dd);
                color: #1f5582;
                border: 2px solid #c0c0c0;
                padding: 20px 15px;
                margin-right: 3px;
                font-size: 24px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 200px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffffff, stop: 1 #f8f9fa);-
                border-bottom: 5px solid #007bff;
                color: #007bff;
                font-weight: 800;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #fff3cd, stop: 1 #ffeaa7);
                color: #856404;
                border-color: #ffc107;
            }
            QPushButton {
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 24px;
                min-height: 32px;
                min-width: 100px;
                text-align: center;
            }
            QPushButton#btn_open {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #28a745, stop: 1 #218838);
                color: white;
                border: 3px solid transparent;
            }
            QPushButton#btn_open:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #34ce57, stop: 1 #28a745);
                border: 3px solid #ffffff;
            }
            QPushButton#btn_open:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #218838, stop: 1 #1e7e34);
            }
            QPushButton#btn_save {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #17a2b8, stop: 1 #138496);
                color: white;
                border: 3px solid transparent;
            }
            QPushButton#btn_save:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #1fc8db, stop: 1 #17a2b8);
                border: 3px solid #ffffff;
            }
            QPushButton#btn_detect {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #dc3545, stop: 1 #c82333);
                color: white;
                border: 2px solid transparent;
                font-size: 24px;
                font-weight: 900;
            }
            QPushButton#btn_detect:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #e74c3c, stop: 1 #dc3545);
                border: 3px solid #ffffff;
            }
            QPushButton#btn_clear {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #6c757d, stop: 1 #5a6268);
                color: white;
                border: 3px solid transparent;
            }
            QPushButton#btn_clear:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #8c959d, stop: 1 #6c757d);
                border: 3px solid #ffffff;
            }
            QPushButton#scenario_btn {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #007bff, stop: 1 #0056b3);
                color: white;
                margin: 0 3px;
                border: 2px solid transparent;
                font-size: 24px;
            }
            QPushButton#scenario_btn:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #0099ff, stop: 1 #007bff);
                border: 3px solid #ffffff;
            }
            QPushButton#scenario_btn:checked {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffc107, stop: 1 #e0a800);
                color: #212529;
                border: 3px solid #fd7e14;
                font-weight: 900;
                font-size: 24px;
            }
            QPushButton#scenario_btn:checked:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffcd39, stop: 1 #ffc107);
            }
            QPushButton#control_btn {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #6f42c1, stop: 1 #5a32a3);
                color: white;
                padding: 6px 12px;
                font-size: 24px;
                border: 1px solid transparent;
            }
            QPushButton#control_btn:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #8e5bd1, stop: 1 #6f42c1);
                border: 2px solid #ffffff;
            }
            QPushButton:disabled {
                background: #e9ecef;
                color: #6c757d;
                border: 2px solid #dee2e6;
            }
            QTreeWidget {
                border: 2px solid #dee2e6;
                background-color: white;
                alternate-background-color: #f8f9fa;
                font-size: 24px;
                selection-background-color: #007bff;
                selection-color: white;
                gridline-color: #dee2e6;
            }
            QTreeWidget::item {
                padding: 8px 6px;
                border-bottom: 1px solid #e9ecef;
                min-height: 20px;
            }
            QTreeWidget::item:selected {
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }
            QTreeWidget::item:hover {
                background-color: #e7f1ff;
                color: #0056b3;
            }
            QHeaderView::section {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #f8f9fa, stop: 1 #e9ecef);
                color: #495057;
                padding: 10px 8px;
                border: 2px solid #dee2e6;
                font-weight: bold;
                font-size: 24px;
            }
            QTextEdit {
                border: 2px solid #dee2e6;
                background-color: white;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 24px;
                line-height: 1.4;
                padding: 8px;
            }
            QComboBox {
                border: 2px solid #dee2e6;
                padding: 6px 10px;
                background-color: white;
                font-size: 24px;
                border-radius: 4px;
            }
            QLabel {
                font-size: 24px;
                color: #495057;
                font-weight: 500;
            }
            QGroupBox {
                font-size: 24px;
                font-weight: bold;
                color: #495057;
                border: 4px solid #dee2e6;
                margin-top: 20px;
                padding-top: 25px;
                border-radius: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 12px 25px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #007bff, stop: 1 #0056b3);
                color: white;
                border: 2px solid #004085;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
            }
            QTableWidget {
                font-size: 24px;
                gridline-color: #dee2e6;
                selection-background-color: #007bff;
                selection-color: white;
                border: 3px solid #dee2e6;
            }
            QTableWidget::item {
                padding: 18px 12px;
                border-bottom: 1px solid #e9ecef;
                min-height: 35px;
            }
            QTableWidget::item:selected {
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }
            QTableWidget::item:hover {
                background-color: #e7f1ff;
                color: #0056b3;
            }
            QProgressBar {
                border: 3px solid #dee2e6;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                min-height: 40px;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #28a745, stop: 1 #20c997);
                border-radius: 8px;
            }
            QFrame {
                border-radius: 8px;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局 - 减少边距
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)
        
        # 创建工具栏
        self.create_toolbar(main_layout)
        
        # 创建主内容区域
        self.create_main_content(main_layout)
        
        # 创建状态栏
        self.create_status_bar()
        
    def create_toolbar(self, parent_layout):
        """创建工具栏"""
        toolbar_frame = QFrame()
        toolbar_frame.setFrameStyle(QFrame.StyledPanel)
        toolbar_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffffff, stop: 1 #f8f9fa);
                border: 3px solid #dee2e6;
                border-radius: 12px;
                margin: 5px;
                padding: 10px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setSpacing(25)  # 增加按钮间距
        toolbar_layout.setContentsMargins(25, 18, 25, 18)  # 增加边距
        
        # 统一按钮尺寸
        button_size = QSize(180, 50)  # 统一按钮大小
        
        # 文件操作按钮
        self.btn_open = QPushButton("📁 打开图像")
        self.btn_open.setObjectName("btn_open")
        self.btn_open.setFixedSize(button_size)
        self.btn_open.clicked.connect(self.open_image)
        
        self.btn_save = QPushButton("💾 保存结果")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setFixedSize(button_size)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        
        # 检测操作按钮
        self.btn_detect = QPushButton("🚀 开始检测")
        self.btn_detect.setObjectName("btn_detect")
        self.btn_detect.setFixedSize(button_size)
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        
        self.btn_clear = QPushButton("🗑️ 清除结果")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.setFixedSize(button_size)
        self.btn_clear.clicked.connect(self.clear_results)
        
        # 应用场景选择 - 改为两个按钮
        scenario_label = QLabel("🎯 应用场景:")
        scenario_label.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        scenario_label.setStyleSheet("""
            color: #007bff;
            background-color: #e7f1ff;
            padding: 12px 18px;
            border-radius: 10px;
            border: 3px solid #b3d7ff;
        """)
        
        # 创建按钮组
        self.scenario_btn_group = QButtonGroup()
        
        self.btn_personal_shopping = QPushButton("🛒 个人购物")
        self.btn_personal_shopping.setObjectName("scenario_btn")
        self.btn_personal_shopping.setFixedSize(button_size)
        self.btn_personal_shopping.setCheckable(True)
        self.btn_personal_shopping.setChecked(True)  # 默认选中
        self.btn_personal_shopping.clicked.connect(lambda: self.set_scenario("personal_shopping"))
        
        self.btn_shelf_audit = QPushButton("📊 货架审计")
        self.btn_shelf_audit.setObjectName("scenario_btn")
        self.btn_shelf_audit.setFixedSize(button_size)
        self.btn_shelf_audit.setCheckable(True)
        self.btn_shelf_audit.clicked.connect(lambda: self.set_scenario("shelf_audit"))
        
        # 添加到按钮组
        self.scenario_btn_group.addButton(self.btn_personal_shopping)
        self.scenario_btn_group.addButton(self.btn_shelf_audit)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(45)
        self.progress_bar.setMinimumWidth(300)
        self.progress_bar.setMaximumHeight(45)
        # 设置进度条样式，确保可见性
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 3px solid #007bff;
                border-radius: 10px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                background-color: #f8f9fa;
                color: #007bff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #28a745, stop: 1 #20c997);
                border-radius: 8px;
            }
        """)
        
        # 布局工具栏组件
        toolbar_layout.addWidget(self.btn_open)
        toolbar_layout.addWidget(self.btn_save)
        
        # 添加分隔线
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("color: #dee2e6; border: 2px solid #dee2e6; margin: 5px;")
        toolbar_layout.addWidget(separator1)
        
        toolbar_layout.addWidget(scenario_label)
        toolbar_layout.addWidget(self.btn_personal_shopping)
        toolbar_layout.addWidget(self.btn_shelf_audit)
        
        # 添加分隔线
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("color: #dee2e6; border: 2px solid #dee2e6; margin: 5px;")
        toolbar_layout.addWidget(separator2)
        
        toolbar_layout.addWidget(self.btn_detect)
        toolbar_layout.addWidget(self.btn_clear)
        
        # 健康分析按钮
        self.btn_health_analysis = QPushButton("🏥 健康分析")
        self.btn_health_analysis.setObjectName("btn_health_analysis")
        self.btn_health_analysis.setFixedSize(button_size)
        self.btn_health_analysis.clicked.connect(self.start_health_analysis)
        self.btn_health_analysis.setEnabled(False)
        toolbar_layout.addWidget(self.btn_health_analysis)

        # 添加分隔线
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setFrameShadow(QFrame.Sunken)
        separator3.setStyleSheet("color: #dee2e6; border: 2px solid #dee2e6; margin: 5px;")
        toolbar_layout.addWidget(separator3)

        # 进度条区域
        progress_label = QLabel("⏳ 处理状态:")
        progress_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        progress_label.setStyleSheet("""
            color: #6c757d;
            padding: 8px 12px;
            border-radius: 6px;
        """)
        toolbar_layout.addWidget(progress_label)
        toolbar_layout.addWidget(self.progress_bar)

        toolbar_layout.addStretch(1)  # 弹性空间
        
        parent_layout.addWidget(toolbar_frame)
        
    def set_scenario(self, scenario):
        """设置当前场景"""
        self.current_scenario = scenario
        
    def create_main_content(self, parent_layout):
        """创建主内容区域"""
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：图像显示区域
        self.create_image_panel(splitter)
        
        # 右侧：结果显示区域
        self.create_results_panel(splitter)
        
        # 设置分割器比例 - 优化布局比例
        splitter.setSizes([800, 600])
        splitter.setHandleWidth(8)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #dee2e6;
                border: 2px solid #adb5bd;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #007bff;
            }
        """)
        
        parent_layout.addWidget(splitter)
        
    def create_image_panel(self, parent_splitter):
        """创建图像显示面板"""
        image_frame = QGroupBox("🖼️ 图像显示")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(10, 30, 10, 10)
        image_layout.setSpacing(10)
        
        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 4px solid #c0c0c0;
                background-color: white;
                min-height: 800px;
                font-size: 24px;
                color: #6c757d;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        self.image_label.setText("📷 请选择图像文件\n\n点击'打开图像'按钮开始")
        
        # 图像控制按钮
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        controls_layout.setContentsMargins(0, 10, 0, 0)
        
        self.btn_zoom_in = QPushButton("🔍+ 放大")
        self.btn_zoom_in.setObjectName("control_btn")
        self.btn_zoom_out = QPushButton("🔍- 缩小")
        self.btn_zoom_out.setObjectName("control_btn")
        self.btn_fit_window = QPushButton("📐 适应窗口")
        self.btn_fit_window.setObjectName("control_btn")
        self.btn_original_size = QPushButton("📏 原始大小")
        self.btn_original_size.setObjectName("control_btn")
        
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit_window.clicked.connect(self.fit_to_window)
        self.btn_original_size.clicked.connect(self.original_size)
        
        controls_layout.addWidget(self.btn_zoom_in)
        controls_layout.addWidget(self.btn_zoom_out)
        controls_layout.addWidget(self.btn_fit_window)
        controls_layout.addWidget(self.btn_original_size)
        controls_layout.addStretch(1)
        
        image_layout.addWidget(self.image_label, 1)  # 给图像标签更多空间
        image_layout.addLayout(controls_layout, 0)   # 控制按钮固定高度
        
        parent_splitter.addWidget(image_frame)
        
    def create_results_panel(self, parent_splitter):
        """创建结果显示面板"""
        results_frame = QGroupBox("📊 检测结果")
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(10, 30, 10, 10)
        results_layout.setSpacing(8)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # 检测结果标签页
        self.create_detection_tab()
        
        # 信息提取标签页
        self.create_extraction_tab()
        
        # OCR文本识别标签页
        self.create_ocr_tab()
        
        # 统计分析标签页
        self.create_analysis_tab()
        
        results_layout.addWidget(self.tab_widget)
        parent_splitter.addWidget(results_frame)
        
    def create_detection_tab(self):
        """创建检测结果标签页"""
        detection_widget = QWidget()
        detection_layout = QVBoxLayout(detection_widget)
        detection_layout.setContentsMargins(8, 8, 8, 8)
        detection_layout.setSpacing(5)
        
        # 创建检测结果树
        self.detection_tree = QTreeWidget()
        self.detection_tree.setHeaderLabels(['ID', '类型', '类别', '置信度', '边界框'])
        self.detection_tree.setAlternatingRowColors(True)
        self.detection_tree.setRootIsDecorated(False)
        self.detection_tree.setItemsExpandable(True)
        
        # 设置列宽
        self.detection_tree.setColumnWidth(0, 80)
        self.detection_tree.setColumnWidth(1, 120)
        self.detection_tree.setColumnWidth(2, 180)
        self.detection_tree.setColumnWidth(3, 120)
        
        # 设置树组件样式
        self.detection_tree.setStyleSheet("""
            QTreeWidget {
                font-size: 22px;
                border: 3px solid #dee2e6;
                border-radius: 8px;
            }
            QTreeWidget::item {
                padding: 18px 12px;
                min-height: 40px;
            }
        """)
        
        detection_layout.addWidget(self.detection_tree)
        
        self.tab_widget.addTab(detection_widget, "🔍 检测结果")
        
    def create_extraction_tab(self):
        """创建信息提取标签页"""
        extraction_widget = QWidget()
        extraction_layout = QVBoxLayout(extraction_widget)
        extraction_layout.setContentsMargins(8, 8, 8, 8)
        extraction_layout.setSpacing(5)
        
        # 创建信息提取文本显示
        self.extraction_text = QTextEdit()
        self.extraction_text.setReadOnly(True)
        self.extraction_text.setFont(QFont("Consolas", 20))
        self.extraction_text.setStyleSheet("""
            QTextEdit {
                font-size: 20px;
                line-height: 1.8;
                padding: 18px;
                border: 3px solid #dee2e6;
                border-radius: 8px;
            }
        """)
        
        extraction_layout.addWidget(self.extraction_text)
        
        self.tab_widget.addTab(extraction_widget, "📄 信息提取")
        
    def create_ocr_tab(self):
        """创建OCR文本识别标签页"""
        ocr_widget = QWidget()
        ocr_layout = QVBoxLayout(ocr_widget)
        ocr_layout.setContentsMargins(8, 8, 8, 8)
        ocr_layout.setSpacing(8)
        
        # OCR结果表格
        self.ocr_table = QTableWidget()
        self.ocr_table.setColumnCount(6)
        self.ocr_table.setHorizontalHeaderLabels(['区域ID', '识别文本', '置信度', 'OCR引擎', '内容特征', '精确位置'])
        self.ocr_table.horizontalHeader().setStretchLastSection(True)
        self.ocr_table.setAlternatingRowColors(True)
        self.ocr_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # 设置列宽
        self.ocr_table.setColumnWidth(0, 100)
        self.ocr_table.setColumnWidth(1, 300)
        self.ocr_table.setColumnWidth(2, 120)
        self.ocr_table.setColumnWidth(3, 150)
        self.ocr_table.setColumnWidth(4, 150)
        
        # 设置表格样式
        self.ocr_table.setStyleSheet("""
            QTableWidget {
                font-size: 20px;
                border: 3px solid #dee2e6;
                border-radius: 8px;
            }
            QTableWidget::item {
                padding: 18px 12px;
                min-height: 45px;
            }
            QHeaderView::section {
                font-size: 20px;
                font-weight: bold;
                padding: 18px 12px;
                min-height: 35px;
            }
        """)
        
        # 文本详细信息
        detail_label = QLabel("📝 文本详细信息:")
        detail_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        detail_label.setStyleSheet("""
            color: #007bff;
            margin-top: 10px;
            margin-bottom: 8px;
            padding: 12px 15px;
            background-color: #e7f1ff;
            border-radius: 8px;
            border: 3px solid #b3d7ff;
        """)
        
        self.ocr_detail_text = QTextEdit()
        self.ocr_detail_text.setReadOnly(True)
        self.ocr_detail_text.setMaximumHeight(360)
        self.ocr_detail_text.setFont(QFont("Consolas", 19))
        self.ocr_detail_text.setStyleSheet("""
            QTextEdit {
                font-size: 19px;
                line-height: 1.7;
                padding: 18px;
                border: 3px solid #dee2e6;
                border-radius: 8px;
            }
        """)
        
        # 连接选择事件
        self.ocr_table.itemSelectionChanged.connect(self.on_ocr_selection_changed)
        
        ocr_layout.addWidget(self.ocr_table, 3)
        ocr_layout.addWidget(detail_label, 0)
        ocr_layout.addWidget(self.ocr_detail_text, 1)
        
        self.tab_widget.addTab(ocr_widget, "🔤 OCR识别")
        
    def create_analysis_tab(self):
        """创建统计分析标签页"""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(15, 15, 15, 15)
        analysis_layout.setSpacing(15)
        
        # 创建统计卡片容器
        stats_container = QFrame()
        stats_container.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 3px solid #dee2e6;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        stats_container_layout = QVBoxLayout(stats_container)
        
        # 统计标题
        stats_title = QLabel("📊 检测统计概览")
        stats_title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        stats_title.setStyleSheet("""
            color: #007bff;
            padding: 15px 0;
            border-bottom: 3px solid #dee2e6;
            margin-bottom: 20px;
        """)
        stats_container_layout.addWidget(stats_title)
        
        # 统计信息网格
        stats_grid = QGridLayout()
        stats_grid.setSpacing(20)
        
        self.stats_labels = {}
        stats_items = [
            ('检测到的商品数量', 'product_count', '🛍️'),
            ('提取的条形码数量', 'barcode_count', '📱'),
            ('文本区域数量', 'text_count', '📝'),
            ('平均检测置信度', 'avg_confidence', '🎯')
        ]
        
        for i, (label_text, key, icon) in enumerate(stats_items):
            # 创建统计卡片
            card_frame = QFrame()
            card_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                               stop: 0 #ffffff, stop: 1 #f8f9fa);
                    border: 2px solid #e9ecef;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px;
                }
                QFrame:hover {
                    border: 2px solid #007bff;
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                               stop: 0 #e7f1ff, stop: 1 #f8f9fa);
                }
            """)
            
            card_layout = QVBoxLayout(card_frame)
            card_layout.setSpacing(8)
            
            # 图标和标签
            icon_label = QLabel(icon)
            icon_label.setFont(QFont("Arial", 36))
            icon_label.setAlignment(Qt.AlignCenter)

            label = QLabel(label_text)
            label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #495057;")

            value_label = QLabel("0")
            value_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setWordWrap(True)  # 启用自动换行
            value_label.setMinimumHeight(80)  # 设置最小高度以容纳多行文本
            value_label.setStyleSheet("""
                color: #007bff;
                background-color: #e7f1ff;
                padding: 10px;
                border-radius: 10px;
                border: 3px solid #b3d7ff;
                line-height: 1.4;
            """)
            
            card_layout.addWidget(icon_label)
            card_layout.addWidget(label)
            card_layout.addWidget(value_label)
            
            row = i // 3
            col = i % 3
            stats_grid.addWidget(card_frame, row, col)
            
            self.stats_labels[key] = value_label
        
        stats_container_layout.addLayout(stats_grid)
        analysis_layout.addWidget(stats_container)
          # 添加弹性空间
        analysis_layout.addStretch(1)
        
        self.tab_widget.addTab(analysis_widget, "📊 统计分析")
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 添加状态栏图标
        try:
            status_icon_path = Path(__file__).parent.parent / "assets" / "icons" / "status_icon_16x16.png"
            if status_icon_path.exists():
                status_icon_label = QLabel()
                status_icon_pixmap = QPixmap(str(status_icon_path))
                status_icon_label.setPixmap(status_icon_pixmap)
                self.status_bar.addPermanentWidget(status_icon_label)
                self.logger.info(f"已设置状态栏图标: {status_icon_path}")
        except Exception as e:
            self.logger.error(f"设置状态栏图标失败: {e}")
        
        self.status_bar.showMessage("智能商品识别系统 - 就绪")
        
    def init_components(self):
        """初始化组件"""
        # 初始化核心组件
        self.image_processor = ImageProcessor()
        self.info_extractor = SimpleInformationExtractor()
        self.text_output_manager = TextOutputManager()
        
        # 在后台线程中初始化检测引擎
        self.init_detection_engine()
        
    def init_detection_engine(self):
        """初始化检测引擎"""
        def init_in_background():
            try:
                self.update_status("正在初始化检测引擎...")
                self.detection_engine = SimpleDetectionEngine()
                self.update_status("检测引擎初始化完成")
                
                # 启用检测按钮
                self.btn_detect.setEnabled(True)
                
            except Exception as e:
                self.logger.error(f"检测引擎初始化失败: {e}")
                self.update_status(f"检测引擎初始化失败: {e}")
                QMessageBox.critical(self, "错误", f"检测引擎初始化失败:\n{e}")
        
        # 在后台线程中初始化
        init_thread = threading.Thread(target=init_in_background, daemon=True)
        init_thread.start()
        
    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.showMessage(message)
        self.logger.info(message)

    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像文件",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.gif);;所有文件 (*.*)"
        )

        if file_path:
            try:
                # 读取图像
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("无法读取图像文件")

                self.current_image_path = file_path

                # 显示图像
                self.display_image(self.current_image)

                # 启用相关按钮
                if self.detection_engine is not None:
                    self.btn_detect.setEnabled(True)

                # 清除之前的结果
                self.clear_results()

                self.update_status(f"已加载图像: {Path(file_path).name}")

            except Exception as e:
                self.logger.error(f"打开图像失败: {e}")
                QMessageBox.critical(self, "错误", f"打开图像失败:\n{e}")

    def display_image(self, image):
        """显示图像"""
        try:
            # 转换颜色空间（OpenCV使用BGR，Qt使用RGB）
            if len(image.shape) == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image

            # 获取图像尺寸
            height, width = display_image.shape[:2]

            # 应用缩放
            if self.zoom_factor != 1.0:
                new_width = int(width * self.zoom_factor)
                new_height = int(height * self.zoom_factor)
                display_image = cv2.resize(display_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                height, width = new_height, new_width

            # 转换为QPixmap - 修复错误的转换方式
            bytes_per_line = 3 * width

            # 确保数据是连续的
            if not display_image.flags['C_CONTIGUOUS']:
                display_image = np.ascontiguousarray(display_image)

            # 正确的转换方式
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # 设置到标签，使用适应大小的方式避免缩放问题
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # 改为True，让图像适应标签大小
            self.image_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            self.logger.error(f"显示图像失败: {e}")
            QMessageBox.warning(self, "警告", f"显示图像失败:\n{e}")

    def zoom_in(self):
        """放大图像"""
        self.zoom_factor *= 1.2
        if self.current_image is not None:
            self.display_image(self.processed_image if self.processed_image is not None else self.current_image)

    def zoom_out(self):
        """缩小图像"""
        self.zoom_factor /= 1.2
        if self.current_image is not None:
            self.display_image(self.processed_image if self.processed_image is not None else self.current_image)

    def fit_to_window(self):
        """适应窗口"""
        if self.current_image is not None:
            # 计算适合窗口的缩放比例
            label_size = self.image_label.size()
            image_size = self.current_image.shape[:2][::-1]  # (width, height)

            scale_x = label_size.width() / image_size[0]
            scale_y = label_size.height() / image_size[1]
            self.zoom_factor = min(scale_x, scale_y) * 0.9  # 留一些边距

            self.display_image(self.processed_image if self.processed_image is not None else self.current_image)

    def original_size(self):
        """原始大小"""
        self.zoom_factor = 1.0
        if self.current_image is not None:
            self.display_image(self.processed_image if self.processed_image is not None else self.current_image)

    def start_detection(self):
        """开始检测"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先选择图像文件")
            return

        if self.detection_engine is None:
            QMessageBox.warning(self, "警告", "检测引擎未初始化")
            return

        try:
            # 禁用检测按钮
            self.btn_detect.setEnabled(False)

            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 不确定进度
            self.progress_bar.setValue(0)
            self.logger.info("进度条已显示")

            # 创建检测工作线程
            self.detection_worker = DetectionWorker(
                self.detection_engine,
                self.info_extractor,
                self.current_image,
                self.current_scenario
            )

            # 连接信号
            self.detection_worker.finished.connect(self.on_detection_finished)
            self.detection_worker.error.connect(self.on_detection_error)
            self.detection_worker.progress.connect(self.update_status)

            # 启动线程
            self.detection_worker.start()

        except Exception as e:
            self.logger.error(f"启动检测失败: {e}")
            QMessageBox.critical(self, "错误", f"启动检测失败:\n{e}")
            self.btn_detect.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_detection_finished(self, detection_results, extraction_results, processed_image):
        """检测完成回调"""
        try:
            # 保存结果
            self.detection_results = detection_results
            self.extraction_results = extraction_results
            self.processed_image = processed_image

            # 显示处理后的图像
            self.display_image(processed_image)

            # 更新各个显示面板
            self.update_detection_display(detection_results)
            self.update_extraction_display(extraction_results, detection_results)
            self.update_ocr_display(extraction_results, detection_results)
            self.update_analysis_display(detection_results, extraction_results)

            # 启用保存按钮和健康分析按钮
            self.btn_save.setEnabled(True)
            self.btn_health_analysis.setEnabled(True)

            self.update_status("检测完成")

        except Exception as e:
            self.logger.error(f"处理检测结果失败: {e}")
            QMessageBox.critical(self, "错误", f"处理检测结果失败:\n{e}")
        finally:
            # 隐藏进度条，启用检测按钮
            self.progress_bar.setVisible(False)
            self.btn_detect.setEnabled(True)
            self.logger.info("进度条已隐藏")

    def on_detection_error(self, error_message):
        """检测错误回调"""
        self.logger.error(f"检测过程出错: {error_message}")
        QMessageBox.critical(self, "检测错误", f"检测过程出错:\n{error_message}")

        # 隐藏进度条，启用检测按钮
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.logger.info("检测错误，进度条已隐藏")

    def update_detection_display(self, detection_results):
        """更新检测结果显示"""
        # 清空树
        self.detection_tree.clear()

        # 添加检测结果
        item_id = 0

        # 添加商品
        if detection_results.get('products'):
            products_item = QTreeWidgetItem(self.detection_tree, ['商品', '', '', '', ''])
            products_item.setExpanded(True)
            for product in detection_results['products']:
                item_id += 1
                bbox_str = f"({product['bbox'][0]}, {product['bbox'][1]}, {product['bbox'][2]}, {product['bbox'][3]})"
                QTreeWidgetItem(products_item, [
                    str(item_id), '商品', product['class_name'],
                    f"{product['confidence']:.3f}", bbox_str
                ])

        # 添加区域
        if detection_results.get('regions'):
            regions_item = QTreeWidgetItem(self.detection_tree, ['区域', '', '', '', ''])
            regions_item.setExpanded(True)
            for region in detection_results['regions']:
                item_id += 1
                bbox_str = f"({region['bbox'][0]}, {region['bbox'][1]}, {region['bbox'][2]}, {region['bbox'][3]})"
                QTreeWidgetItem(regions_item, [
                    str(item_id), '区域', region['class_name'],
                    f"{region['confidence']:.3f}", bbox_str
                ])


        # 添加文本
        if detection_results.get('texts'):
            texts_item = QTreeWidgetItem(self.detection_tree, ['文本', '', '', '', ''])
            texts_item.setExpanded(True)
            for text in detection_results['texts']:
                item_id += 1
                bbox_str = f"({text['bbox'][0]}, {text['bbox'][1]}, {text['bbox'][2]}, {text['bbox'][3]})"
                QTreeWidgetItem(texts_item, [
                    str(item_id), '文本', text['class_name'],
                    f"{text['confidence']:.3f}", bbox_str
                ])

    def update_extraction_display(self, extraction_results, detection_results=None):
        """更新信息提取显示"""
        content = "=== 信息提取结果 ===\n\n"

        # 条形码信息
        if extraction_results.get('barcodes'):
            content += "📊 条形码信息:\n"
            for i, barcode in enumerate(extraction_results['barcodes'], 1):
                content += f"  {i}. 类型: {barcode['type']}\n"
                content += f"     数据: {barcode['data']}\n"
                content += f"     置信度: {barcode['confidence']:.3f}\n\n"

        # 营养信息
        if extraction_results.get('nutrition_info'):
            content += "🥗 营养成分信息:\n"
            nutrition = extraction_results['nutrition_info']
            for key, value in nutrition.items():
                if key != 'raw_texts' and value is not None:
                    content += f"  {key}: {value}\n"
            content += "\n"



        # 使用YOLO区域进行精确OCR文本识别结果
        if extraction_results.get('text_info'):
            content += "📝 基于YOLO区域的精确OCR识别结果:\n"
            content += "=" * 50 + "\n"
            
            for i, text in enumerate(extraction_results['text_info'], 1):
                region_id = text.get('region_id', i)
                content += f"🔍 区域 {region_id}:\n"
                content += f"   📝 识别文本: {text['text']}\n"
                content += f"   🎯 OCR置信度: {text['confidence']:.3f}\n"
                content += f"   🔧 OCR引擎: {text.get('engine', 'unknown')}\n"

                # 显示YOLO检测信息
                yolo_region = text.get('yolo_region')
                if yolo_region:
                    content += f"   🎯 YOLO检测信息:\n"
                    content += f"      检测类别: {yolo_region.get('class_name', 'unknown')}\n"
                    content += f"      检测置信度: {yolo_region.get('confidence', 0):.3f}\n"

                # 显示位置信息
                if 'bbox' in text:
                    bbox = text['bbox']
                    content += f"   📐 原图位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n"
                
                region_bbox = text.get('region_bbox')
                if region_bbox:
                    x1, y1, x2, y2 = region_bbox
                    width = x2 - x1
                    height = y2 - y1
                    content += f"   📏 区域尺寸: {width}×{height}像素 (面积: {width * height})\n"

                content += f"   📊 文本长度: {len(text['text'])} 字符\n"
                
                # 文本质量评估
                confidence = text['confidence']
                if confidence > 0.8:
                    content += f"   ✅ 识别质量: 优秀\n"
                elif confidence > 0.6:
                    content += f"   ⚠️ 识别质量: 良好\n"
                else:
                    content += f"   ❌ 识别质量: 较差，建议人工检查\n"
                
                content += "\n"
            
            # 添加提取方法说明
            extraction_summary = extraction_results.get('extraction_summary', {})
            ocr_method = extraction_summary.get('ocr_extraction_method', 'unknown')
            yolo_regions_count = extraction_summary.get('yolo_text_regions', 0)
            
            content += f"📋 提取方法说明:\n"
            if ocr_method == 'yolo_regions':
                content += f"   ✅ 使用 {yolo_regions_count} 个YOLO检测区域进行精确OCR提取\n"
                content += f"   🎯 提取精度: 高 (基于AI检测的精确区域)\n"
            else:
                content += f"   ⚠️ 使用全图OCR提取 (未检测到文本区域)\n"
                content += f"   🎯 提取精度: 中等 (全图扫描)\n"
            content += "\n"

        # YOLO文本检测结果
        if detection_results and detection_results.get('texts'):
            content += "🔍 YOLO文本区域检测:\n"
            for i, text_region in enumerate(detection_results['texts'], 1):
                content += f"  {i}. 检测区域: {text_region['class_name']}\n"
                content += f"     置信度: {text_region['confidence']:.3f}\n"
                bbox = text_region['bbox']
                content += f"     位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n"

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                content += f"     区域大小: {width}×{height} (面积: {area})\n\n"

        # 提取摘要
        if extraction_results.get('extraction_summary'):
            content += "📋 提取摘要:\n"
            summary = extraction_results['extraction_summary']
            content += f"  条形码数量: {summary.get('barcode_count', 0)}\n"
            content += f"  包含营养信息: {'是' if summary.get('has_nutrition_info', False) else '否'}\n"
            content += f"  文本区域数量: {summary.get('text_regions_count', 0)}\n"

        self.extraction_text.setPlainText(content)

    def update_ocr_display(self, extraction_results, detection_results=None):
        """更新OCR文本识别显示"""
        # 清空表格
        self.ocr_table.setRowCount(0)

        # 添加从YOLO区域精确提取的OCR结果
        ocr_texts = extraction_results.get('text_info', [])
        row = 0

        for text_info in ocr_texts:
            self.ocr_table.insertRow(row)

            text_content = text_info['text']
            confidence = text_info['confidence']
            engine = text_info.get('engine', 'unknown')
            region_id = text_info.get('region_id', row + 1)

            # 分析文本特征
            features = self._analyze_text_features(text_content)

            # 格式化位置信息
            bbox = text_info.get('bbox', (0, 0, 0, 0))
            position = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"

            # 如果有YOLO区域信息，显示更详细的信息
            yolo_region = text_info.get('yolo_region')
            if yolo_region:
                yolo_conf = yolo_region.get('confidence', 0)
                display_text = f"[YOLO置信度:{yolo_conf:.3f}] {text_content}"
                engine_info = f"{engine} (YOLO区域)"
            else:
                display_text = text_content
                engine_info = engine

            # 设置表格项
            self.ocr_table.setItem(row, 0, QTableWidgetItem(f"区域{region_id}"))
            self.ocr_table.setItem(row, 1, QTableWidgetItem(display_text))
            self.ocr_table.setItem(row, 2, QTableWidgetItem(f"{confidence:.3f}"))
            self.ocr_table.setItem(row, 3, QTableWidgetItem(engine_info))
            self.ocr_table.setItem(row, 4, QTableWidgetItem(features))
            self.ocr_table.setItem(row, 5, QTableWidgetItem(position))

            # 根据置信度设置行颜色
            if confidence > 0.8:
                color = "#d4edda"  # 绿色背景，高置信度
            elif confidence > 0.6:
                color = "#fff3cd"  # 黄色背景，中等置信度
            else:
                color = "#f8d7da"  # 红色背景，低置信度

            for col in range(6):
                item = self.ocr_table.item(row, col)
                if item:
                    item.setBackground(QColor(color))

            row += 1

        # 如果没有OCR结果但有YOLO检测区域，显示检测区域信息
        if not ocr_texts and detection_results and detection_results.get('texts'):
            for i, text_region in enumerate(detection_results['texts']):
                self.ocr_table.insertRow(row)

                bbox = text_region['bbox']
                position = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                size_info = f"{width}×{height}px"

                self.ocr_table.setItem(row, 0, QTableWidgetItem(f"检测区域{i+1}"))
                self.ocr_table.setItem(row, 1, QTableWidgetItem(f"[未提取到文本] {text_region['class_name']}"))
                self.ocr_table.setItem(row, 2, QTableWidgetItem(f"{text_region['confidence']:.3f}"))
                self.ocr_table.setItem(row, 3, QTableWidgetItem('YOLO检测'))
                self.ocr_table.setItem(row, 4, QTableWidgetItem(size_info))
                self.ocr_table.setItem(row, 5, QTableWidgetItem(position))

                # 设置灰色背景表示未提取到文本
                for col in range(6):
                    item = self.ocr_table.item(row, col)
                    if item:
                        item.setBackground(QColor("#e9ecef"))

                row += 1

        # 如果完全没有结果，显示提示信息
        if row == 0:
            self.ocr_table.insertRow(0)
            self.ocr_table.setItem(0, 0, QTableWidgetItem("暂无"))
            self.ocr_table.setItem(0, 1, QTableWidgetItem("未检测到文本区域或未识别到文本"))
            self.ocr_table.setItem(0, 2, QTableWidgetItem("--"))
            self.ocr_table.setItem(0, 3, QTableWidgetItem("--"))
            self.ocr_table.setItem(0, 4, QTableWidgetItem("--"))
            self.ocr_table.setItem(0, 5, QTableWidgetItem("--"))

        # 调整列宽
        self.ocr_table.resizeColumnsToContents()

    def _analyze_text_features(self, text):
        """分析文本特征"""
        if not text:
            return "空"

        features = []

        if any(c.isdigit() for c in text):
            features.append("数字")
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            features.append("中文")
        if any(c.isalpha() and ord(c) < 128 for c in text):
            features.append("英文")

        special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?")
        if any(c in special_chars for c in text):
            features.append("符号")

        return ", ".join(features) if features else "其他"

    def on_ocr_selection_changed(self):
        """OCR选择变化事件"""
        try:
            current_row = self.ocr_table.currentRow()
            if current_row >= 0:
                # 获取选中行的数据
                region_item = self.ocr_table.item(current_row, 0)
                text_item = self.ocr_table.item(current_row, 1)
                confidence_item = self.ocr_table.item(current_row, 2)
                engine_item = self.ocr_table.item(current_row, 3)
                features_item = self.ocr_table.item(current_row, 4)
                position_item = self.ocr_table.item(current_row, 5)

                if text_item:
                    # 显示详细信息
                    detail_info = f"🔍 选中文本区域详细信息:\n"
                    detail_info += "=" * 50 + "\n\n"
                    
                    detail_info += f"📍 区域标识: {region_item.text() if region_item else 'N/A'}\n"
                    detail_info += f"📝 识别文本: {text_item.text()}\n"
                    detail_info += f"🎯 OCR置信度: {confidence_item.text() if confidence_item else 'N/A'}\n"
                    detail_info += f"🔧 识别引擎: {engine_item.text() if engine_item else 'N/A'}\n"
                    detail_info += f"🏷️ 内容特征: {features_item.text() if features_item else 'N/A'}\n"
                    detail_info += f"📐 精确位置: {position_item.text() if position_item else 'N/A'}\n\n"

                    # 如果是从YOLO区域提取的文本，显示更多详细信息
                    if engine_item and 'YOLO区域' in engine_item.text():
                        # 尝试从extraction_results中获取对应的详细信息
                        if hasattr(self, 'extraction_results') and self.extraction_results:
                            text_infos = self.extraction_results.get('text_info', [])
                            if current_row < len(text_infos):
                                text_info = text_infos[current_row]
                                yolo_region = text_info.get('yolo_region')
                                if yolo_region:
                                    detail_info += "🎯 YOLO检测信息:\n"
                                    detail_info += f"  检测类别: {yolo_region.get('class_name', 'unknown')}\n"
                                    detail_info += f"  检测置信度: {yolo_region.get('confidence', 0):.3f}\n"
                                    detail_info += f"  检测类型: {yolo_region.get('type', 'text')}\n"
                                    
                                    region_bbox = text_info.get('region_bbox')
                                    if region_bbox:
                                        x1, y1, x2, y2 = region_bbox
                                        width = x2 - x1
                                        height = y2 - y1
                                        detail_info += f"  区域尺寸: {width}×{height}像素\n"
                                        detail_info += f"  区域面积: {width * height}平方像素\n"
                                    detail_info += "\n"

                    # 如果是OCR结果，显示文本分析
                    if engine_item and 'YOLO' not in engine_item.text():
                        text_content = text_item.text()
                        # 清理显示文本（去除YOLO置信度信息）
                        clean_text = text_content
                        if clean_text.startswith('[YOLO置信度:'):
                            clean_text = clean_text.split('] ', 1)[-1] if '] ' in clean_text else clean_text
                        
                        detail_info += "📊 文本分析:\n"
                        detail_info += f"  字符总数: {len(clean_text)}\n"
                        detail_info += f"  单词数量: {len(clean_text.split())}\n"
                        detail_info += f"  行数: {len(clean_text.splitlines())}\n"

                        # 字符类型统计
                        char_stats = {
                            '数字': sum(1 for c in clean_text if c.isdigit()),
                            '英文字母': sum(1 for c in clean_text if c.isalpha() and ord(c) < 128),
                            '中文字符': sum(1 for c in clean_text if '\u4e00' <= c <= '\u9fff'),
                            '空格': sum(1 for c in clean_text if c.isspace()),
                            '标点符号': sum(1 for c in clean_text if not (c.isdigit() or c.isalpha() or c.isspace())),
                        }

                        detail_info += "\n  字符类型分布:\n"
                        for char_type, count in char_stats.items():
                            if count > 0:
                                percentage = (count / len(clean_text)) * 100 if clean_text else 0
                                detail_info += f"    {char_type}: {count} ({percentage:.1f}%)\n"

                        # 文本质量评估
                        detail_info += "\n  文本质量评估:\n"
                        if text_info.get('confidence', 0) > 0.8:
                            detail_info += "    ✅ 高质量识别结果\n"
                        elif text_info.get('confidence', 0) > 0.6:
                            detail_info += "    ⚠️ 中等质量识别结果\n"
                        else:
                            detail_info += "    ❌ 低质量识别结果，建议人工检查\n"

                    self.ocr_detail_text.setPlainText(detail_info)

        except Exception as e:
            self.logger.error(f"OCR选择事件处理失败: {e}")
            self.ocr_detail_text.setPlainText(f"❌ 显示详细信息时出错: {str(e)}")

    def update_analysis_display(self, detection_results, extraction_results):
        """更新统计分析显示"""
        try:
            # 计算统计数据
            products = detection_results.get('products', [])
            barcodes = extraction_results.get('barcodes', [])
            texts = detection_results.get('texts', [])
            
            # 统计商品名称和数量
            product_names = {}
            for product in products:
                # 优先使用映射后的名称，如果没有则使用原始名称
                name = product.get('class_name', '未知商品')
                if name in product_names:
                    product_names[name] += 1
                else:
                    product_names[name] = 1
            
            # 生成商品显示文本
            if product_names:
                product_display = "\n".join([f"{name}: {count}个" for name, count in product_names.items()])
            else:
                product_display = "0"
            

            # 计算平均置信度
            all_confidences = []
            for category in ['products', 'regions', 'texts']:
                for item in detection_results.get(category, []):
                    all_confidences.append(item['confidence'])

            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

            # 更新显示
            self.stats_labels['product_count'].setText(product_display)
            self.stats_labels['barcode_count'].setText(str(len(barcodes)))
            self.stats_labels['text_count'].setText(str(len(texts)))
            self.stats_labels['avg_confidence'].setText(f"{avg_confidence:.3f}")

        except Exception as e:
            self.logger.error(f"更新统计分析失败: {e}")

    def save_results(self):
        """保存结果"""
        if self.detection_results is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        try:
            # 选择保存目录
            save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
            if not save_dir:
                return

            save_path = Path(save_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []

            # 保存处理后的图像
            if self.processed_image is not None:
                image_path = save_path / f"detection_result_{timestamp}.jpg"
                cv2.imwrite(str(image_path), self.processed_image)
                saved_files.append(f"检测结果图像: {image_path.name}")

            # 保存检测结果JSON
            results_path = save_path / f"detection_data_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                # 转换numpy数组为列表以便JSON序列化
                serializable_results = self._make_json_serializable(self.detection_results)
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            saved_files.append(f"检测数据JSON: {results_path.name}")

            # 使用文本输出管理器保存多种格式的文本结果
            if self.extraction_results:
                # 保存纯文本结果
                text_file = self.text_output_manager.save_text_results(
                    self.extraction_results, self.current_image_path
                )
                if text_file:
                    saved_files.append(f"文本识别结果: {Path(text_file).name}")

                # 保存CSV格式
                csv_file = self.text_output_manager.save_csv_results(
                    self.extraction_results, self.current_image_path
                )
                if csv_file:
                    saved_files.append(f"CSV数据: {Path(csv_file).name}")

                # 保存完整JSON结果
                json_file = self.text_output_manager.save_json_results(
                    self.extraction_results, self.detection_results, self.current_image_path
                )
                if json_file:
                    saved_files.append(f"完整JSON: {Path(json_file).name}")

                # 生成汇总报告
                summary_file = self.text_output_manager.generate_summary_report(
                    self.extraction_results, self.detection_results, self.current_image_path
                )
                if summary_file:
                    saved_files.append(f"汇总报告: {Path(summary_file).name}")

            # 显示保存成功信息
            success_message = f"结果已保存到: {save_path}\n\n保存的文件:\n" + "\n".join(f"• {file}" for file in saved_files)
            QMessageBox.information(self, "保存成功", success_message)
            self.update_status(f"结果已保存到: {save_path} ({len(saved_files)}个文件)")

        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            QMessageBox.critical(self, "错误", f"保存结果失败:\n{e}")

    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    def clear_results(self):
        """清除结果"""
        # 清空检测结果树
        self.detection_tree.clear()

        # 清空信息提取显示
        self.extraction_text.clear()

        # 清空OCR显示
        self.ocr_table.setRowCount(0)
        self.ocr_detail_text.clear()

        # 重置统计显示
        for label in self.stats_labels.values():
            label.setText("0")

        # 重置图像显示
        if self.current_image is not None:
            self.display_image(self.current_image)

        # 清除结果数据
        self.detection_results = None
        self.extraction_results = None
        self.processed_image = None

        # 禁用保存按钮和健康分析按钮
        self.btn_save.setEnabled(False)
        self.btn_health_analysis.setEnabled(False)
        
        # 清除健康分析结果
        self.health_analysis_results = None

        self.update_status("结果已清除")

    def start_health_analysis(self):
        """开始健康分析"""
        if not self.extraction_results:
            QMessageBox.warning(self, "警告", "请先进行图像检测和OCR识别")
            return
            
        try:
            # 初始化Gemini分析器
            if not self.gemini_analyzer:
                self.gemini_analyzer = GeminiHealthAnalyzer()
                
            # 检查API可用性
            if not self.gemini_analyzer.is_available():
                QMessageBox.critical(self, "错误", "Gemini API不可用，请检查API密钥配置")
                return
                
            # 提取OCR文本和坐标信息
            text_data = []
            if 'text_info' in self.extraction_results:
                for text_info in self.extraction_results['text_info']:
                    text_data.append({
                        'text': text_info.get('text', ''),
                        'bbox': text_info.get('bbox', []),
                        'confidence': text_info.get('confidence', 0)
                    })
                    
            # 启动健康分析线程
            self.health_worker = HealthAnalysisWorker(self.gemini_analyzer, text_data, self.current_image_path)
            self.health_worker.finished.connect(self.on_health_analysis_finished)
            self.health_worker.error.connect(self.on_health_analysis_error)
            
            # 禁用按钮，显示进度
            self.btn_health_analysis.setEnabled(False)
            self.update_status("正在进行健康分析...")
            
            self.health_worker.start()
            
        except Exception as e:
            self.logger.error(f"启动健康分析失败: {e}")
            QMessageBox.critical(self, "错误", f"启动健康分析失败:\n{e}")
            
    def on_health_analysis_finished(self, analysis_results):
        """健康分析完成回调"""
        try:
            self.health_analysis_results = analysis_results
            
            # 显示分析结果
            self.show_health_analysis_results(analysis_results)
            
            self.update_status("健康分析完成")
            
        except Exception as e:
            self.logger.error(f"处理健康分析结果失败: {e}")
            QMessageBox.critical(self, "错误", f"处理健康分析结果失败:\n{e}")
        finally:
            self.btn_health_analysis.setEnabled(True)
            
    def on_health_analysis_error(self, error_message):
        """健康分析错误回调"""
        self.logger.error(f"健康分析出错: {error_message}")
        QMessageBox.critical(self, "健康分析错误", f"健康分析出错:\n{error_message}")
        self.btn_health_analysis.setEnabled(True)
        
    def show_health_analysis_results(self, results):
        """显示健康分析结果"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🏥 健康分析结果")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # 创建文本显示区域
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        
        # 格式化显示结果
        formatted_text = "🏥 产品健康分析报告\n"
        formatted_text += "=" * 50 + "\n\n"
        
        if 'overall_score' in results and results['overall_score'] is not None:
            score = results['overall_score']
            score_emoji = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
            formatted_text += f"📊 总体健康评分: {score_emoji} {score}/10\n\n"
        
        # OCR错误纠正
        if 'ocr_corrections' in results and results['ocr_corrections']:
            formatted_text += "🔍 OCR错误纠正:\n"
            for correction in results['ocr_corrections']:
                if isinstance(correction, dict):
                    original = correction.get('original', '')
                    corrected = correction.get('corrected', '')
                    formatted_text += f"  • '{original}' → '{corrected}'\n"
                else:
                    formatted_text += f"  • {correction}\n"
            formatted_text += "\n"
        
        # 检测到的成分
        if 'detected_ingredients' in results and results['detected_ingredients']:
            formatted_text += "🧪 检测到的成分:\n"
            for ingredient in results['detected_ingredients']:
                formatted_text += f"  • {ingredient}\n"
            formatted_text += "\n"
        
        # 营养成分
        if 'nutrition_facts' in results and results['nutrition_facts']:
            formatted_text += "📊 营养成分信息:\n"
            for key, value in results['nutrition_facts'].items():
                formatted_text += f"  • {key}: {value}\n"
            formatted_text += "\n"
            
        if 'analysis' in results and results['analysis']:
            formatted_text += f"📝 详细分析:\n{results['analysis']}\n\n"
            
        if 'recommendations' in results and results['recommendations']:
            formatted_text += "💡 建议:\n"
            if isinstance(results['recommendations'], list):
                for rec in results['recommendations']:
                    formatted_text += f"  • {rec}\n"
            else:
                formatted_text += f"{results['recommendations']}\n"
            formatted_text += "\n"
        
        # 健康警告
        if 'health_warnings' in results and results['health_warnings']:
            formatted_text += "⚠️ 健康警告:\n"
            for warning in results['health_warnings']:
                formatted_text += f"  • {warning}\n"
                
        text_area.setPlainText(formatted_text)
        layout.addWidget(text_area)
        
        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()

class HealthAnalysisWorker(QThread):
    """健康分析工作线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, gemini_analyzer, text_data, image_path):
        super().__init__()
        self.gemini_analyzer = gemini_analyzer
        self.text_data = text_data
        self.image_path = image_path
        
    def run(self):
        try:
            # 调用Gemini进行健康分析
            results = self.gemini_analyzer.analyze_product_health(self.text_data, self.image_path)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("智能商品识别系统")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Smart Product Analysis")
    
    # 设置应用程序级别的图标（用于任务栏）
    try:
        # 获取ICO文件路径（Windows任务栏）
        ico_path = Path(__file__).parent.parent / "assets" / "icons" / "app_icon_32x32.ico"
        if ico_path.exists():
            app_icon = QIcon(str(ico_path))
        else:
            # 回退到PNG文件
            png_path = Path(__file__).parent.parent / "assets" / "icons" / "app_icon_32x32.png"
            if png_path.exists():
                app_icon = QIcon(str(png_path))
            else:
                app_icon = None
        
        if app_icon:
            app.setWindowIcon(app_icon)
            print(f"✓ 已设置应用程序图标")
    except Exception as e:
        print(f"⚠ 设置应用程序图标失败: {e}")

    # 创建主窗口
    window = PyQt5MainWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
