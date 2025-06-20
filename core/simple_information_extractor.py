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
from .chinese_ocr_optimizer import ChineseOCROptimizer

class SimpleInformationExtractor:
    """简化的信息提取器，避免PaddleOCR初始化问题"""
    
    def __init__(self):
        """初始化简化信息提取器"""
        self.logger = logging.getLogger(__name__)
        self.image_processor = ImageProcessor()
        
        # 初始化中文OCR优化器
        try:
            self.chinese_ocr_optimizer = ChineseOCROptimizer()
            chinese_ocr_available = True
        except Exception as e:
            self.logger.warning(f"中文OCR优化器初始化失败: {e}")
            self.chinese_ocr_optimizer = None
            chinese_ocr_available = False
        
        self.logger.info("初始化简化信息提取器...")
        
        # 检查可用的功能
        self.features = {
            'barcode': PYZBAR_AVAILABLE,
            'tesseract_ocr': PYTESSERACT_AVAILABLE,
            'chinese_ocr_optimization': chinese_ocr_available,
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
                
                # 确保边界框在图像范围内并添加边距
                h, w = image.shape[:2]
                margin = 5  # 添加5像素边距
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin) 
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                # 裁剪文本区域
                text_region_img = image[y1:y2, x1:x2]
                
                if text_region_img.size == 0:
                    self.logger.warning(f"文本区域 {i+1} 为空，跳过")
                    continue
                
                # 优先使用中文OCR优化器
                if self.chinese_ocr_optimizer and self.features['chinese_ocr_optimization']:
                    try:
                        # 使用优化器进行高精度识别
                        optimized_results = self.chinese_ocr_optimizer.optimize_chinese_text_recognition(
                            text_region_img, region
                        )
                        
                        if optimized_results:
                            # 选择最佳结果
                            best_result = max(optimized_results, key=lambda x: x['confidence'])
                            
                            # 调整坐标到原图坐标系
                            adjusted_bbox = [
                                best_result['bbox'][0] + x1,
                                best_result['bbox'][1] + y1,
                                best_result['bbox'][2] + x1,
                                best_result['bbox'][3] + y1
                            ]
                            
                            result = {
                                'region_id': i + 1,
                                'text': best_result['text'],
                                'bbox': bbox,  # YOLO检测的原始边界框
                                'precise_bbox': adjusted_bbox,  # OCR精确定位的边界框
                                'region_bbox': (x1, y1, x2, y2),  # 裁剪区域在原图中的位置
                                'confidence': best_result['confidence'],
                                'engine': 'chinese_ocr_optimizer',
                                'config_type': best_result.get('config_type', 'unknown'),
                                'chinese_char_count': best_result.get('chinese_char_count', 0),
                                'scale_used': best_result.get('scale_used', 'original'),
                                'yolo_region': region  # 保存原始YOLO检测信息
                            }
                            
                            # 如果有原始文本，也保存
                            if 'original_text' in best_result:
                                result['original_text'] = best_result['original_text']
                            
                            extracted_texts.append(result)
                            self.logger.info(f"区域 {i+1} 优化识别: '{best_result['text']}' (置信度: {best_result['confidence']:.3f})")
                            continue
                        
                    except Exception as e:
                        self.logger.warning(f"中文OCR优化器处理区域 {i+1} 失败: {e}，回退到传统方法")
                
                # 回退到传统OCR方法
                fallback_result = self._fallback_text_recognition(
                    text_region_img, region, i, bbox, (x1, y1, x2, y2)
                )
                
                if fallback_result:
                    extracted_texts.append(fallback_result)
                    
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
                # 检查是否为条形码区域（支持中文和英文）
                class_name = region['class_name'].lower()
                if 'barcode' in class_name or '条形码' in class_name:
                    region_image = self.image_processor.crop_region(image, region['bbox'])
                    barcodes = self.extract_barcode(region_image)
                    comprehensive_info['barcodes'].extend(barcodes)
                    self.logger.info(f"在条形码区域 '{region['class_name']}' 中检测到 {len(barcodes)} 个条形码")
            
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
    
    def _fallback_text_recognition(self, text_region_img: np.ndarray, region: Dict, 
                                 region_index: int, bbox: Tuple, region_bbox: Tuple) -> Optional[Dict]:
        """
        回退文本识别方法（当优化器失败时使用）
        
        Args:
            text_region_img: 文本区域图像
            region: 区域信息
            region_index: 区域索引
            bbox: 边界框
            region_bbox: 区域边界框
            
        Returns:
            识别结果或None
        """
        try:
            # 预处理文本区域图像
            processed_img = self._preprocess_text_region(text_region_img)
            
            # 判断是否为营养成分表区域
            is_nutrition = self._is_nutrition_table_region(region)
            
            # 选择合适的OCR配置
            if is_nutrition:
                # 营养成分表专用配置
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.%克毫微千焦卡能量蛋白质脂肪碳水化合物糖钠钙铁锌维生素膳食纤维胆固醇反式饱和不参考值含量营养素成分表每份NRV'
            else:
                # 普通文本配置
                custom_config = r'--oem 3 --psm 6'
            
            # 提取文本
            text = pytesseract.image_to_string(
                processed_img,
                lang='chi_sim+eng',
                config=custom_config
            ).strip()
            
            if text:
                # 获取详细的OCR结果，包括置信度和坐标
                try:
                    data = pytesseract.image_to_data(
                        processed_img,
                        config=custom_config,
                        lang='chi_sim+eng',
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # 计算平均置信度
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.5
                    
                    # 获取文本的精确边界框
                    text_boxes = []
                    for i in range(len(data['text'])):
                        if data['text'][i].strip() and int(data['conf'][i]) > 30:
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            text_boxes.append([x, y, x + w, y + h])
                    
                    # 计算整体文本边界框
                    if text_boxes:
                        min_x = min(box[0] for box in text_boxes)
                        min_y = min(box[1] for box in text_boxes)
                        max_x = max(box[2] for box in text_boxes)
                        max_y = max(box[3] for box in text_boxes)
                        
                        # 映射到原图坐标
                        precise_bbox = [
                            min_x + region_bbox[0],
                            min_y + region_bbox[1],
                            max_x + region_bbox[0],
                            max_y + region_bbox[1]
                        ]
                    else:
                        precise_bbox = list(bbox)
                    
                except Exception as e:
                    self.logger.warning(f"获取详细OCR数据失败: {e}")
                    avg_confidence = 0.5
                    precise_bbox = list(bbox)
                
                # 构建结果
                result = {
                    'region_id': region_index + 1,
                    'text': text,
                    'bbox': bbox,  # YOLO检测的原始边界框
                    'precise_bbox': precise_bbox,  # OCR精确定位的边界框
                    'region_bbox': region_bbox,  # 裁剪区域在原图中的位置
                    'confidence': avg_confidence,
                    'engine': 'tesseract_fallback',
                    'config_type': 'nutrition' if is_nutrition else 'default',
                    'yolo_region': region  # 保存原始YOLO检测信息
                }
                
                self.logger.info(f"区域 {region_index+1} 回退识别: '{text}' (置信度: {avg_confidence:.3f})")
                return result
            else:
                self.logger.warning(f"区域 {region_index+1} 回退方法未识别到文本")
                return None
                
        except Exception as e:
            self.logger.error(f"回退文本识别失败: {e}")
            return None
    
    def _is_nutrition_table_region(self, region: Dict) -> bool:
        """
        判断检测到的区域是否为营养成分表
        
        Args:
            region: 区域信息字典
            
        Returns:
            是否为营养成分表区域
        """
        try:
            # 检查类别名称
            class_name = region.get('class_name', '').lower()
            nutrition_keywords = ['nutrition', 'table', '营养', '成分', '表格']
            
            for keyword in nutrition_keywords:
                if keyword in class_name:
                    return True
            
            # 检查区域大小和宽高比（营养成分表通常是矩形且面积较大）
            bbox = region.get('bbox', [])
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 0
                
                # 营养成分表通常宽高比在0.5-2.0之间，且面积较大
                area = width * height
                if 0.5 <= aspect_ratio <= 2.0 and area > 10000:
                    return True
                    
            return False
             
        except Exception as e:
            self.logger.error(f"判断营养成分表区域失败: {e}")
            return False
    
    def get_feature_status(self) -> Dict:
        """获取功能状态"""
        return {
            'barcode_extraction': self.features['barcode'],
            'text_extraction': self.features['tesseract_ocr'],
            'chinese_ocr_optimization': self.features['chinese_ocr_optimization'],
            'image_processing': self.features['image_processing'],
            'available_engines': [
                'pyzbar' if self.features['barcode'] else None,
                'tesseract' if self.features['tesseract_ocr'] else None,
                'chinese_ocr_optimizer' if self.features['chinese_ocr_optimization'] else None
            ]
        }
