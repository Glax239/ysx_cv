# -*- coding: utf-8 -*-
"""
简化的信息提取模块
Simplified Information Extraction Module
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re

# 条形码相关导入
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

# OCR相关导入
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from .image_processor import ImageProcessor

class SimpleInformationExtractor:
    """简化的信息提取器，避免PaddleOCR初始化问题"""
    
    def __init__(self):
        """初始化简化信息提取器"""
        self.logger = logging.getLogger(__name__)
        self.image_processor = ImageProcessor()
        self.logger.info("初始化简化信息提取器...")
        
        # 检查可用的功能
        self.features = {
            'barcode': PYZBAR_AVAILABLE,
            'tesseract_ocr': PYTESSERACT_AVAILABLE,
            'image_processing': True
        }
        
        self.logger.info(f"可用功能: {self.features}")
    
    def extract_barcode(self, image: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        提取条形码信息
        
        Args:
            image: 输入图像
            preprocess: 是否进行预处理
            
        Returns:
            条形码信息列表
        """
        if not PYZBAR_AVAILABLE:
            self.logger.warning("pyzbar未安装，无法进行条形码识别")
            return []
        
        try:
            processed_image = image
            
            if preprocess:
                # 使用图像处理器增强条形码区域
                processed_image = self.image_processor.enhance_barcode_region(image)
            
            # 使用pyzbar解码条形码
            barcodes = pyzbar.decode(processed_image)
            
            results = []
            for barcode in barcodes:
                # 提取条形码数据
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                # 获取边界框
                rect = barcode.rect
                bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                
                # 获取多边形顶点
                polygon = [(point.x, point.y) for point in barcode.polygon]
                
                result = {
                    'data': barcode_data,
                    'type': barcode_type,
                    'bbox': bbox,
                    'polygon': polygon,
                    'confidence': 1.0  # pyzbar不提供置信度，设为1.0
                }
                
                results.append(result)
                self.logger.info(f"检测到条形码: {barcode_type} - {barcode_data}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"条形码提取失败: {e}")
            return []
    
    def extract_text_simple(self, image: np.ndarray) -> List[Dict]:
        """
        使用简单方法提取文本信息
        
        Args:
            image: 输入图像
            
        Returns:
            文本信息列表
        """
        if not PYTESSERACT_AVAILABLE:
            self.logger.warning("pytesseract未安装，无法进行OCR识别")
            return []
        
        try:
            # 预处理图像
            processed_image = self.image_processor.enhance_text_region(image)
            
            # 检测文本方向并校正
            angle = self.image_processor.detect_text_orientation(processed_image)
            if abs(angle) > 1:
                processed_image = self.image_processor.rotate_image(processed_image, angle)
            
            # 使用Tesseract提取文本
            try:
                text = pytesseract.image_to_string(processed_image, lang='chi_sim+eng')
                
                if text.strip():
                    return [{
                        'text': text.strip(),
                        'bbox': (0, 0, image.shape[1], image.shape[0]),
                        'confidence': 0.8,  # 默认置信度
                        'engine': 'tesseract_simple'
                    }]
                else:
                    return []
                    
            except Exception as e:
                self.logger.warning(f"Tesseract OCR失败: {e}")
                return []
            
        except Exception as e:
            self.logger.error(f"文本提取失败: {e}")
            return []
    
    def extract_text_from_regions(self, image: np.ndarray, text_regions: List[Dict]) -> List[Dict]:
        """
        从YOLO检测的文本区域中精确提取文本
        
        Args:
            image: 原始图像
            text_regions: YOLO检测到的文本区域列表
            
        Returns:
            精确提取的文本信息列表
        """
        if not PYTESSERACT_AVAILABLE:
            self.logger.warning("pytesseract未安装，无法进行OCR识别")
            return []
        
        extracted_texts = []
        
        for i, region in enumerate(text_regions):
            try:
                # 获取文本区域的边界框
                bbox = region['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # 确保边界框在图像范围内
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1) 
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # 裁剪文本区域
                text_region_img = image[y1:y2, x1:x2]
                
                if text_region_img.size == 0:
                    self.logger.warning(f"文本区域 {i+1} 为空，跳过")
                    continue
                
                # 预处理文本区域图像以提高OCR准确性
                processed_img = self._preprocess_text_region(text_region_img)
                
                # 使用Tesseract进行OCR识别
                # 优化OCR参数以提高中文识别准确性
                # 移除字符白名单限制，允许识别所有中文字符
                custom_config = r'--oem 3 --psm 6'

                # 提取文本 - 优先使用中文简体，然后是英文
                text = pytesseract.image_to_string(
                    processed_img,
                    lang='chi_sim+eng',  # 调整语言优先级，中文优先
                    config=custom_config
                ).strip()
                
                if text:
                    # 获取详细的OCR结果，包括置信度
                    data = pytesseract.image_to_data(
                        processed_img,
                        config=custom_config,
                        lang='chi_sim+eng',  # 保持与上面一致的语言优先级
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # 计算平均置信度
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.5
                    
                    # 构建结果
                    result = {
                        'region_id': i + 1,
                        'text': text,
                        'bbox': bbox,  # 原图中的位置
                        'region_bbox': (x1, y1, x2, y2),  # 裁剪区域在原图中的位置
                        'confidence': avg_confidence,
                        'engine': 'tesseract_region',
                        'yolo_region': region  # 保存原始YOLO检测信息
                    }
                    
                    extracted_texts.append(result)
                    self.logger.info(f"区域 {i+1} 识别文本: '{text}' (置信度: {avg_confidence:.3f})")
                else:
                    self.logger.warning(f"区域 {i+1} 未识别到文本")
                    
            except Exception as e:
                self.logger.error(f"处理文本区域 {i+1} 时出错: {e}")
                continue
        
        return extracted_texts
    
    def _preprocess_text_region(self, region_img: np.ndarray) -> np.ndarray:
        """
        预处理文本区域以提高OCR准确性
        
        Args:
            region_img: 文本区域图像
            
        Returns:
            预处理后的图像
        """
        try:
            # 如果图像太小，先放大
            h, w = region_img.shape[:2]
            if h < 30 or w < 30:
                scale_factor = max(2, 40 / min(h, w))
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                region_img = cv2.resize(region_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 转换为灰度图
            if len(region_img.shape) == 3:
                gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = region_img
            
            # 应用高斯滤波去噪
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 形态学操作，去除小噪点
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 再次放大以提高OCR精度
            if cleaned.shape[0] < 50:
                scale_factor = 2
                new_h = int(cleaned.shape[0] * scale_factor)
                new_w = int(cleaned.shape[1] * scale_factor)
                cleaned = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"文本区域预处理失败: {e}")
            return region_img
    
    def extract_comprehensive_info(self, image: np.ndarray, detection_results: Dict) -> Dict:
        """
        简化的综合信息提取
        
        Args:
            image: 原始图像
            detection_results: 检测结果
            
        Returns:
            综合提取的信息
        """
        comprehensive_info = {
            'barcodes': [],
            'nutrition_info': {},
            'text_info': [],
            'extraction_summary': {}
        }
        
        try:
            # 提取条形码信息（从检测到的区域）
            for region in detection_results.get('regions', []):
                if 'barcode' in region['class_name'].lower():
                    region_image = self.image_processor.crop_region(image, region['bbox'])
                    barcodes = self.extract_barcode(region_image)
                    comprehensive_info['barcodes'].extend(barcodes)
            
            # 如果没有检测到条形码区域，尝试全图检测
            if not comprehensive_info['barcodes']:
                barcodes = self.extract_barcode(image)
                comprehensive_info['barcodes'].extend(barcodes)
            
            # 使用YOLO检测到的文本区域进行精确OCR提取
            text_regions = detection_results.get('texts', [])
            if text_regions:
                self.logger.info(f"发现 {len(text_regions)} 个YOLO文本区域，开始精确OCR提取...")
                extracted_texts = self.extract_text_from_regions(image, text_regions)
                comprehensive_info['text_info'].extend(extracted_texts)
            else:
                self.logger.info("未发现YOLO文本区域，尝试全图OCR提取...")
                # 如果没有检测到文本区域，尝试简单的全图OCR
                texts = self.extract_text_simple(image)
                comprehensive_info['text_info'].extend(texts)
            

            
            # 简化的营养信息解析
            all_texts = [text['text'] for text in comprehensive_info['text_info']]
            if all_texts:
                nutrition_info = self._parse_nutrition_simple(all_texts)
                comprehensive_info['nutrition_info'] = nutrition_info
            
            # 生成提取摘要
            comprehensive_info['extraction_summary'] = {
                'barcode_count': len(comprehensive_info['barcodes']),
                'has_nutrition_info': bool(comprehensive_info['nutrition_info']),
                'text_regions_count': len(comprehensive_info['text_info']),
                'yolo_text_regions': len(text_regions),
                'ocr_extraction_method': 'yolo_regions' if text_regions else 'full_image'
            }
            
            self.logger.info("综合信息提取完成")
            return comprehensive_info
            
        except Exception as e:
            self.logger.error(f"综合信息提取失败: {e}")
            return comprehensive_info
    
    def _parse_nutrition_simple(self, texts: List[str]) -> Dict:
        """简化的营养信息解析"""
        nutrition_info = {
            'energy': None,
            'protein': None,
            'fat': None,
            'carbohydrate': None,
            'sodium': None,
            'sugar': None,
            'raw_texts': texts
        }
        
        # 合并所有文本
        full_text = ' '.join(texts).lower()
        
        # 简化的营养成分匹配模式
        patterns = {
            'energy': [r'(\d+(?:\.\d+)?)\s*(?:kj|千焦|kcal|卡路里)', r'能量[:\s]*(\d+(?:\.\d+)?)'],
            'protein': [r'蛋白质[:\s]*(\d+(?:\.\d+)?)', r'protein[:\s]*(\d+(?:\.\d+)?)'],
            'fat': [r'脂肪[:\s]*(\d+(?:\.\d+)?)', r'fat[:\s]*(\d+(?:\.\d+)?)'],
            'sodium': [r'钠[:\s]*(\d+(?:\.\d+)?)', r'sodium[:\s]*(\d+(?:\.\d+)?)'],
            'sugar': [r'糖[:\s]*(\d+(?:\.\d+)?)', r'sugar[:\s]*(\d+(?:\.\d+)?)']
        }
        
        # 匹配各种营养成分
        for nutrient, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    try:
                        nutrition_info[nutrient] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        return nutrition_info
    
    def get_feature_status(self) -> Dict:
        """获取功能状态"""
        return {
            'barcode_extraction': self.features['barcode'],
            'text_extraction': self.features['tesseract_ocr'],
            'image_processing': self.features['image_processing'],
            'available_engines': [
                'pyzbar' if self.features['barcode'] else None,
                'tesseract' if self.features['tesseract_ocr'] else None
            ]
        }
