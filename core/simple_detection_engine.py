# -*- coding: utf-8 -*-
"""
简化的检测引擎
Simplified Detection Engine for faster initialization and testing
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

from config import MODEL_PATHS, DETECTION_CONFIG
from utils.name_mapper import get_global_mapper

class SimpleDetectionEngine:
    """简化的检测引擎，专注于YOLO检测，避免PaddleOCR初始化问题"""
    
    def __init__(self):
        """初始化简化检测引擎"""
        self.models = {}
        self.class_colors = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化简化检测引擎...")
        self._load_models()
        self._generate_colors()
        self.logger.info("简化检测引擎初始化完成")
    
    def _load_models(self):
        """加载YOLO模型"""
        try:
            for model_name, model_path in MODEL_PATHS.items():
                if Path(model_path).exists():
                    self.logger.info(f"正在加载模型: {model_name}")
                    start_time = time.time()
                    
                    # 使用更安全的模型加载方式
                    try:
                        model = YOLO(model_path)
                        # 预热模型
                        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                        _ = model.predict(dummy_img, verbose=False, conf=0.1)
                        
                        self.models[model_name] = model
                        load_time = time.time() - start_time
                        self.logger.info(f"成功加载模型 {model_name} (耗时: {load_time:.2f}秒)")
                        
                    except Exception as e:
                        self.logger.warning(f"加载模型 {model_name} 失败: {e}")
                        continue
                else:
                    self.logger.warning(f"模型文件不存在: {model_path}")
                    
        except Exception as e:
            self.logger.error(f"加载模型时出错: {e}")
            raise
    
    def _generate_colors(self):
        """为每个模型的类别生成颜色"""
        np.random.seed(42)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'names'):
                num_classes = len(model.names)
                colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')
                self.class_colors[model_name] = colors
    
    def detect_with_model(self, image: np.ndarray, model_name: str) -> List[Dict]:
        """
        使用指定模型进行检测
        
        Args:
            image: 输入图像
            model_name: 模型名称
            
        Returns:
            检测结果列表
        """
        if model_name not in self.models:
            self.logger.warning(f"模型 {model_name} 未加载")
            return []
        
        try:
            self.logger.info(f"使用模型 {model_name} 进行检测...")
            start_time = time.time()
            
            results = self.models[model_name].predict(
                image,
                conf=DETECTION_CONFIG['confidence_threshold'],
                iou=DETECTION_CONFIG['iou_threshold'],
                max_det=DETECTION_CONFIG['max_det'],
                device=DETECTION_CONFIG['device'],
                verbose=DETECTION_CONFIG['verbose'],
                half=DETECTION_CONFIG['half'],
                imgsz=DETECTION_CONFIG['imgsz']
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        original_class_name = self.models[model_name].names[class_id]
                        
                        # 应用名称映射
                        name_mapper = get_global_mapper()
                        mapped_class_name = name_mapper.map_class_name(
                            model_name, class_id, original_class_name
                        )
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': mapped_class_name,
                            'original_class_name': original_class_name,  # 保留原始名称
                            'type': model_name.replace('_detector', ''),
                            'model': model_name
                        })
            
            detect_time = time.time() - start_time
            self.logger.info(f"模型 {model_name} 检测完成，检测到 {len(detections)} 个对象 (耗时: {detect_time:.2f}秒)")
            return detections
            
        except Exception as e:
            self.logger.error(f"使用模型 {model_name} 检测时出错: {e}")
            return []
    
    def comprehensive_detection(self, image: np.ndarray) -> Dict:
        """
        综合检测：使用所有可用模型进行检测
        
        Args:
            image: 输入图像
            
        Returns:
            包含所有检测结果的字典
        """
        self.logger.info("开始综合检测...")
        start_time = time.time()
        
        results = {
            'products': [],
            'regions': [],
            'texts': [],
            'hierarchical_results': [],
            'all_detections': []
        }
        
        try:
            # 检测商品
            if 'product_detector' in self.models:
                products = self.detect_with_model(image, 'product_detector')
                results['products'] = products
                results['all_detections'].extend(products)
            
            # 检测区域
            if 'region_detector' in self.models:
                regions = self.detect_with_model(image, 'region_detector')
                results['regions'] = regions
                results['all_detections'].extend(regions)
            

            
            # 检测文本
            if 'text_detector' in self.models:
                texts = self.detect_with_model(image, 'text_detector')
                results['texts'] = texts
                results['all_detections'].extend(texts)
            
            total_time = time.time() - start_time
            total_detections = len(results['all_detections'])
            
            self.logger.info(f"综合检测完成，总共检测到 {total_detections} 个对象 (总耗时: {total_time:.2f}秒)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"综合检测过程中出错: {e}")
            return results
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       draw_labels: bool = True, thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            draw_labels: 是否绘制标签
            thickness: 边框粗细
            
        Returns:
            绘制了检测结果的图像
        """
        result_image = image.copy()
        
        # 定义颜色映射
        color_map = {
            'product': (0, 255, 0),      # 绿色
            'region': (255, 0, 0),       # 蓝色
            'text': (255, 255, 0),       # 青色
            'cocotext': (0, 255, 255)    # 黄色
        }
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            detection_type = detection.get('type', 'unknown')
            
            x1, y1, x2, y2 = bbox
            
            # 选择颜色
            color = color_map.get(detection_type, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            if draw_labels:
                # 准备标签文本
                label = f'{class_name} {confidence:.2f}'
                
                # 计算文本尺寸
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                # 绘制标签背景
                label_y1 = max(y1 - text_height - baseline, 0)
                label_y2 = y1
                cv2.rectangle(result_image, (x1, label_y1), 
                            (x1 + text_width, label_y2), color, -1)
                
                # 绘制标签文本
                cv2.putText(result_image, label, (x1, y1 - baseline),
                          font, font_scale, (255, 255, 255), font_thickness)
        
        return result_image
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {}
        for model_name, model in self.models.items():
            info[model_name] = {
                'loaded': True,
                'classes': list(model.names.values()) if hasattr(model, 'names') else [],
                'num_classes': len(model.names) if hasattr(model, 'names') else 0
            }
        
        # 添加未加载的模型信息
        for model_name in MODEL_PATHS.keys():
            if model_name not in self.models:
                info[model_name] = {
                    'loaded': False,
                    'classes': [],
                    'num_classes': 0
                }
        
        return info
