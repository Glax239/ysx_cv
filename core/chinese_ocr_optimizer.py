# -*- coding: utf-8 -*-
"""
中文OCR优化器
Chinese OCR Optimizer

专门用于提高中文文本识别准确率和坐标定位精度
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
import re
import pytesseract
from pytesseract import Output
from dataclasses import dataclass

from config import OCR_CONFIG, IMAGE_PROCESSING_CONFIG
from .enhanced_image_processor import EnhancedImageProcessor

class ChineseOCROptimizer:
    """中文OCR优化器，提供高精度的中文文本识别和坐标定位"""
    
    def __init__(self):
        """初始化中文OCR优化器"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化增强图像处理器
        self.image_processor = EnhancedImageProcessor()
        
        # 中文字符正则表达式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        # 营养成分相关关键词
        self.nutrition_keywords = {
            '能量', '蛋白质', '脂肪', '碳水化合物', '糖', '钠', '钙', '铁', '锌',
            '维生素', '膳食纤维', '胆固醇', '反式脂肪', '饱和脂肪', '不饱和脂肪',
            '千焦', '千卡', '克', '毫克', '微克', '营养成分表', '每100克', '每份',
            '参考值', 'NRV', '%', '含量', '营养素'
        }
        
        # 常见OCR错误映射
        self.error_corrections = {
            # 数字错误
            'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 's': '5',
            # 中文错误
            '白质': '蛋白质', '碳水': '碳水化合物', '膳食': '膳食纤维',
            '维生': '维生素', '千焦': '千焦', '毫克': '毫克',
            # 单位错误
            'g': '克', 'mg': '毫克', 'ug': '微克', 'kJ': '千焦', 'kcal': '千卡'
        }
        
        # 多种OCR配置
        self.ocr_configs = {
            'default': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u4e00-\u9fff%().,：:：、。，',
            'nutrition': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.%克毫微千焦卡能量蛋白质脂肪碳水化合物糖钠钙铁锌维生素膳食纤维胆固醇反式饱和不参考值含量营养素成分表每份NRV',
            'numbers_only': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.%',
            'chinese_only': '--oem 3 --psm 6 -c tessedit_char_whitelist=\u4e00-\u9fff',
            'mixed_text': '--oem 3 --psm 6',
            'table_structure': '--oem 3 --psm 6 -c tessedit_create_hocr=1',
            'dense_text': '--oem 3 --psm 4',
            'sparse_text': '--oem 3 --psm 3',
            'chinese_enhanced': '--oem 3 --psm 6 -c preserve_interword_spaces=1'
        }
        
        # 多种缩放比例
        self.scale_factors = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        
        # 错误纠正字典
        self.correction_dict = {
            # 常见OCR错误
            '能量': ['能量', '能最', '能童', '龙量', '能蛋'],
            '蛋白质': ['蛋白质', '蛋白贡', '蛋自质', '蛋臼质', '蛋曰质'],
            '脂肪': ['脂肪', '脂防', '指肪', '旨肪', '脂访'],
            '碳水化合物': ['碳水化合物', '碳水化台物', '炭水化合物', '碳水化舍物', '碳水化含物'],
            '营养': ['营养', '营美', '营羞', '菅养', '营义'],
            '成分': ['成分', '成份', '戍分', '戌分', '戊分'],
            '含量': ['含量', '含童', '舍量', '舍童', '含蛋'],
            '参考值': ['参考值', '参考直', '叁考值', '叁考直', '参考植'],
            '每100克': ['每100克', '每100g', '每1009', '每100g', '每100G'],
            '千焦': ['千焦', '干焦', '千蕉', '干蕉', '千集'],
            '毫克': ['毫克', '毫g', '亳克', '亳g', '毫G'],
            '微克': ['微克', '微g', '徽克', '徽g', '微G'],
            '钠': ['钠', '纳', '钠', '呐'],
            '钙': ['钙', '该', '钙', '盖'],
            '铁': ['铁', '鉄', '铁', '贴'],
            '锌': ['锌', '辛', '锌', '新'],
            '维生素': ['维生素', '维生紊', '维生索', '维生素'],
            '膳食纤维': ['膳食纤维', '膳食纤雏', '膳食纤维', '膳食织维']
        }
        
        self.logger.info("中文OCR优化器初始化完成")
    
    def optimize_chinese_text_recognition(self, image: np.ndarray, region: Dict) -> List[Dict]:
        """
        优化中文文本识别，提供高精度的文本和坐标信息
        
        Args:
            image: 输入图像
            region: 区域信息
            
        Returns:
            优化后的识别结果列表
        """
        try:
            # 1. 检测并纠正文本方向
            angle = self.image_processor.detect_text_orientation(image)
            if abs(angle) > 1.0:  # 只有角度足够大才纠正
                corrected_image = self.image_processor.correct_skew(image, angle)
                self.logger.info(f"检测到倾斜角度: {angle:.2f}°，已纠正")
            else:
                corrected_image = image
            
            # 2. 多尺度预处理
            processed_images = self._multi_scale_preprocessing(corrected_image)
            
            # 3. 多配置识别
            all_results = []
            
            for scale_name, processed_img in processed_images.items():
                for config_name, config in self.ocr_configs.items():
                    try:
                        # 获取详细OCR结果
                        ocr_data = pytesseract.image_to_data(
                            processed_img,
                            lang=OCR_CONFIG['languages'],
                            config=config,
                            output_type=Output.DICT
                        )
                        
                        # 解析OCR结果并获取精确坐标
                        parsed_results = self._parse_ocr_data_with_coordinates(
                            ocr_data, scale_name, config_name, processed_img.shape
                        )
                        
                        all_results.extend(parsed_results)
                        
                    except Exception as e:
                        self.logger.warning(f"配置 {config_name} 在尺度 {scale_name} 下识别失败: {e}")
                        continue
            
            # 4. 结果融合和优化
            optimized_results = self._merge_and_optimize_results(all_results, region)
            
            # 5. 坐标映射回原图
            final_results = self._map_coordinates_to_original(optimized_results, image.shape, region)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"中文OCR优化失败: {e}")
            return []
    
    def _multi_scale_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        多尺度图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            不同尺度的预处理图像字典
        """
        processed_images = {}
        
        try:
            # 原始尺寸
            processed_images['original'] = self._enhance_image_for_ocr(image, scale=1.0)
            
            # 1.5倍放大
            processed_images['scale_1_5'] = self._enhance_image_for_ocr(image, scale=1.5)
            
            # 2倍放大
            processed_images['scale_2_0'] = self._enhance_image_for_ocr(image, scale=2.0)
            
            # 3倍放大（适合小字体）
            processed_images['scale_3_0'] = self._enhance_image_for_ocr(image, scale=3.0)
            
            return processed_images
            
        except Exception as e:
            self.logger.error(f"多尺度预处理失败: {e}")
            return {'original': image}
    
    def _enhance_image_for_ocr(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        为OCR优化图像
        
        Args:
            image: 输入图像
            scale: 缩放倍数
            
        Returns:
            优化后的图像
        """
        try:
            # 1. 缩放图像
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 2. 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 3. CLAHE对比度增强
            clahe = cv2.createCLAHE(
                clipLimit=IMAGE_PROCESSING_CONFIG.get('clahe_clip_limit', 2.0),
                tileGridSize=(8, 8)
            )
            enhanced = clahe.apply(gray)
            
            # 4. 轻微去噪
            denoised = cv2.medianBlur(enhanced, 3)
            
            # 5. 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                IMAGE_PROCESSING_CONFIG.get('adaptive_threshold_block_size', 11),
                IMAGE_PROCESSING_CONFIG.get('adaptive_threshold_c', 2)
            )
            
            # 6. 轻微形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 7. 边缘锐化
            kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(cleaned, -1, kernel_sharpen)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"图像OCR优化失败: {e}")
            return image
    
    def _parse_ocr_data_with_coordinates(self, ocr_data: Dict, scale_name: str, 
                                       config_name: str, image_shape: Tuple) -> List[Dict]:
        """
        解析OCR数据并获取精确坐标信息
        
        Args:
            ocr_data: Tesseract OCR数据
            scale_name: 尺度名称
            config_name: 配置名称
            image_shape: 图像尺寸
            
        Returns:
            解析后的结果列表
        """
        results = []
        
        try:
            # 获取文本级别的数据
            n_boxes = len(ocr_data['level'])
            
            for i in range(n_boxes):
                # 只处理单词级别的数据（level 5）
                if ocr_data['level'][i] == 5:
                    text = ocr_data['text'][i].strip()
                    conf = int(ocr_data['conf'][i])
                    
                    # 过滤低置信度和空文本
                    if conf > OCR_CONFIG.get('confidence_threshold', 30) and text:
                        # 获取边界框坐标
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        
                        # 验证坐标有效性
                        if w > 0 and h > 0 and x >= 0 and y >= 0:
                            # 计算中文字符数量
                            chinese_chars = len(self.chinese_pattern.findall(text))
                            
                            result = {
                                'text': text,
                                'bbox': [x, y, x + w, y + h],
                                'confidence': conf / 100.0,
                                'scale_name': scale_name,
                                'config_name': config_name,
                                'chinese_char_count': chinese_chars,
                                'word_index': i,
                                'image_shape': image_shape
                            }
                            
                            results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"解析OCR数据失败: {e}")
            return []
    
    def _merge_and_optimize_results(self, all_results: List[Dict], region: Dict) -> List[Dict]:
        """
        合并和优化识别结果
        
        Args:
            all_results: 所有识别结果
            region: 区域信息
            
        Returns:
            优化后的结果
        """
        if not all_results:
            return []
        
        try:
            # 1. 按文本内容分组
            text_groups = {}
            for result in all_results:
                text = result['text']
                if text not in text_groups:
                    text_groups[text] = []
                text_groups[text].append(result)
            
            # 2. 为每个文本选择最佳结果
            best_results = []
            for text, group in text_groups.items():
                # 选择置信度最高的结果
                best_result = max(group, key=lambda x: x['confidence'])
                
                # 应用文本后处理
                processed_text = self._post_process_chinese_text(text, region)
                best_result['text'] = processed_text
                best_result['original_text'] = text
                
                best_results.append(best_result)
            
            # 3. 按置信度排序
            best_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return best_results
            
        except Exception as e:
            self.logger.error(f"结果合并优化失败: {e}")
            return all_results
    
    def _post_process_chinese_text(self, text: str, region: Dict) -> str:
        """
        中文文本后处理
        
        Args:
            text: 原始文本
            region: 区域信息
            
        Returns:
            处理后的文本
        """
        try:
            # 1. 基本清理
            cleaned_text = text.strip()
            
            # 2. 应用错误纠正
            for error, correction in self.error_corrections.items():
                cleaned_text = cleaned_text.replace(error, correction)
            
            # 3. 营养成分表特殊处理
            if self._is_nutrition_region(region):
                cleaned_text = self._process_nutrition_text(cleaned_text)
            
            # 4. 去除多余的空格和特殊字符
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z%().，。：:、\s]', '', cleaned_text)
            
            return cleaned_text.strip()
            
        except Exception as e:
            self.logger.error(f"文本后处理失败: {e}")
            return text
    
    def _is_nutrition_region(self, region: Dict) -> bool:
        """
        判断是否为营养成分表区域
        
        Args:
            region: 区域信息
            
        Returns:
            是否为营养成分表
        """
        try:
            class_name = region.get('class_name', '').lower()
            return 'nutrition' in class_name or '营养' in class_name
        except:
            return False
    
    def _process_nutrition_text(self, text: str) -> str:
        """
        处理营养成分表文本
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        try:
            # 营养成分表特殊处理
            # 1. 修正常见营养成分名称
            nutrition_corrections = {
                '白质': '蛋白质',
                '碳水': '碳水化合物',
                '膳食': '膳食纤维',
                '维生': '维生素',
                '能量': '能量',
                '脂肪': '脂肪'
            }
            
            for error, correction in nutrition_corrections.items():
                text = text.replace(error, correction)
            
            # 2. 数字和单位处理
            # 确保数字和单位之间有适当的间隔
            text = re.sub(r'(\d+)([a-zA-Z\u4e00-\u9fff])', r'\1 \2', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"营养成分文本处理失败: {e}")
            return text
    
    def _map_coordinates_to_original(self, results: List[Dict], original_shape: Tuple, 
                                   region: Dict) -> List[Dict]:
        """
        将坐标映射回原图
        
        Args:
            results: 识别结果
            original_shape: 原图尺寸
            region: 区域信息
            
        Returns:
            映射后的结果
        """
        try:
            mapped_results = []
            
            for result in results:
                # 获取缩放信息
                scale_name = result.get('scale_name', 'original')
                image_shape = result.get('image_shape', original_shape)
                
                # 计算缩放比例
                if scale_name == 'scale_1_5':
                    scale_factor = 1.5
                elif scale_name == 'scale_2_0':
                    scale_factor = 2.0
                elif scale_name == 'scale_3_0':
                    scale_factor = 3.0
                else:
                    scale_factor = 1.0
                
                # 获取原始边界框
                bbox = result['bbox']
                
                # 映射坐标
                if scale_factor != 1.0:
                    # 先映射回原始尺寸
                    mapped_bbox = [
                        int(bbox[0] / scale_factor),
                        int(bbox[1] / scale_factor),
                        int(bbox[2] / scale_factor),
                        int(bbox[3] / scale_factor)
                    ]
                else:
                    mapped_bbox = bbox
                
                # 如果有区域偏移，需要加上区域在原图中的位置
                region_bbox = region.get('bbox', [0, 0, original_shape[1], original_shape[0]])
                if len(region_bbox) >= 4:
                    final_bbox = [
                        mapped_bbox[0] + region_bbox[0],
                        mapped_bbox[1] + region_bbox[1],
                        mapped_bbox[2] + region_bbox[0],
                        mapped_bbox[3] + region_bbox[1]
                    ]
                else:
                    final_bbox = mapped_bbox
                
                # 确保坐标在图像范围内
                h, w = original_shape[:2]
                final_bbox = [
                    max(0, min(w, final_bbox[0])),
                    max(0, min(h, final_bbox[1])),
                    max(0, min(w, final_bbox[2])),
                    max(0, min(h, final_bbox[3]))
                ]
                
                # 创建最终结果
                mapped_result = {
                    'text': result['text'],
                    'bbox': final_bbox,
                    'confidence': result['confidence'],
                    'config_type': result.get('config_name', 'unknown'),
                    'chinese_char_count': result.get('chinese_char_count', 0),
                    'original_text': result.get('original_text', result['text']),
                    'scale_used': scale_name,
                    'engine': 'chinese_ocr_optimizer'
                }
                
                mapped_results.append(mapped_result)
            
            return mapped_results
            
        except Exception as e:
            self.logger.error(f"坐标映射失败: {e}")
            return results