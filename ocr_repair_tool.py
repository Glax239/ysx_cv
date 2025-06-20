#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR修复工具
实时修复OCR识别中的坐标和文本问题
"""

import cv2
import numpy as np
import os
import sys
import logging
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import pytesseract
import re
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.simple_information_extractor import SimpleInformationExtractor
from core.enhanced_image_processor import EnhancedImageProcessor
from core.chinese_ocr_optimizer import ChineseOCROptimizer

@dataclass
class RepairResult:
    """修复结果数据类"""
    success: bool
    original_text: str
    repaired_text: str
    original_bbox: List[float]
    repaired_bbox: List[float]
    confidence_improvement: float
    repair_methods: List[str]
    processing_time: float

class OCRRepairTool:
    """
    OCR修复工具
    专门修复OCR识别中的常见问题
    """
    
    def __init__(self):
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.extractor = SimpleInformationExtractor()
        self.image_processor = EnhancedImageProcessor()
        self.chinese_optimizer = ChineseOCROptimizer()
        
        # 修复配置
        self.repair_config = {
            'coordinate_expansion': 5,  # 坐标扩展像素
            'min_confidence_threshold': 0.3,
            'max_repair_attempts': 3,
            'enable_chinese_correction': True,
            'enable_number_correction': True,
            'enable_coordinate_refinement': True
        }
        
        # 文本修复规则
        self.text_repair_rules = {
            # 中文常见错误
            'chinese_corrections': {
                '能最': '能量',
                '蛋白贡': '蛋白质',
                '脂防': '脂肪',
                '碳水化台物': '碳水化合物',
                '碳水化舍物': '碳水化合物',
                '营美': '营养',
                '戍分': '成分',
                '含童': '含量',
                '参考直': '参考值',
                '毫兑': '毫克',
                '千焦': '千焦',
                '干焦': '千焦',
                '项目': '项目',
                '每100兑': '每100克',
                '每100g': '每100克'
            },
            # 数字常见错误
            'number_corrections': {
                'O': '0',
                'l': '1',
                'I': '1',
                'S': '5',
                'G': '6',
                'B': '8',
                'g': '9'
            },
            # 单位修正
            'unit_corrections': {
                'g': '克',
                'mg': '毫克',
                'kj': '千焦',
                'kcal': '千卡',
                '%': '%'
            }
        }
        
        # 坐标修复策略
        self.coordinate_strategies = [
            'expand_bbox',
            'refine_with_contours',
            'multi_scale_detection',
            'adaptive_threshold_detection'
        ]
    
    def repair_ocr_result(self, image: np.ndarray, ocr_result: Dict) -> Dict[str, Any]:
        """
        修复OCR识别结果
        
        Args:
            image: 输入图像
            ocr_result: OCR识别结果
            
        Returns:
            修复后的结果
        """
        try:
            start_time = time.time()
            
            text_regions = ocr_result.get('text_regions', [])
            repaired_regions = []
            repair_summary = {
                'total_regions': len(text_regions),
                'repaired_regions': 0,
                'repair_methods_used': [],
                'average_confidence_improvement': 0.0
            }
            
            confidence_improvements = []
            
            for i, region in enumerate(text_regions):
                repair_result = self._repair_single_region(image, region, i)
                
                if repair_result.success:
                    # 更新区域信息
                    repaired_region = region.copy()
                    repaired_region['text'] = repair_result.repaired_text
                    repaired_region['bbox'] = repair_result.repaired_bbox
                    repaired_region['confidence'] = region.get('confidence', 0) + repair_result.confidence_improvement
                    repaired_region['repair_info'] = {
                        'original_text': repair_result.original_text,
                        'repair_methods': repair_result.repair_methods,
                        'confidence_improvement': repair_result.confidence_improvement
                    }
                    
                    repaired_regions.append(repaired_region)
                    repair_summary['repaired_regions'] += 1
                    repair_summary['repair_methods_used'].extend(repair_result.repair_methods)
                    confidence_improvements.append(repair_result.confidence_improvement)
                else:
                    repaired_regions.append(region)
            
            # 计算平均置信度改进
            if confidence_improvements:
                repair_summary['average_confidence_improvement'] = np.mean(confidence_improvements)
            
            # 构建修复后的结果
            repaired_result = ocr_result.copy()
            repaired_result['text_regions'] = repaired_regions
            repaired_result['repair_summary'] = repair_summary
            repaired_result['repair_time'] = time.time() - start_time
            
            self.logger.info(f"修复完成: {repair_summary['repaired_regions']}/{repair_summary['total_regions']} 个区域")
            
            return repaired_result
            
        except Exception as e:
            self.logger.error(f"OCR结果修复失败: {e}")
            return ocr_result
    
    def _repair_single_region(self, image: np.ndarray, region: Dict, region_id: int) -> RepairResult:
        """
        修复单个文本区域
        """
        start_time = time.time()
        
        original_text = region.get('text', '').strip()
        original_bbox = region.get('bbox', [])
        original_confidence = region.get('confidence', 0)
        
        repair_methods = []
        repaired_text = original_text
        repaired_bbox = original_bbox.copy() if original_bbox else []
        confidence_improvement = 0.0
        
        try:
            # 1. 文本修复
            if original_text:
                text_repair_result = self._repair_text_content(original_text)
                if text_repair_result['changed']:
                    repaired_text = text_repair_result['text']
                    repair_methods.extend(text_repair_result['methods'])
                    confidence_improvement += 0.1  # 文本修复带来的置信度提升
            
            # 2. 坐标修复
            if len(original_bbox) >= 4 and self.repair_config['enable_coordinate_refinement']:
                coord_repair_result = self._repair_coordinates(image, original_bbox, original_text)
                if coord_repair_result['changed']:
                    repaired_bbox = coord_repair_result['bbox']
                    repair_methods.extend(coord_repair_result['methods'])
                    confidence_improvement += coord_repair_result['confidence_boost']
            
            # 3. 如果置信度仍然很低，尝试重新识别
            if (original_confidence + confidence_improvement) < self.repair_config['min_confidence_threshold']:
                rerecognition_result = self._rerecognize_region(image, repaired_bbox)
                if rerecognition_result['success']:
                    repaired_text = rerecognition_result['text']
                    confidence_improvement += rerecognition_result['confidence_boost']
                    repair_methods.append('rerecognition')
            
            # 判断修复是否成功
            success = len(repair_methods) > 0 or confidence_improvement > 0
            
            return RepairResult(
                success=success,
                original_text=original_text,
                repaired_text=repaired_text,
                original_bbox=original_bbox,
                repaired_bbox=repaired_bbox,
                confidence_improvement=confidence_improvement,
                repair_methods=repair_methods,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"区域 {region_id} 修复失败: {e}")
            return RepairResult(
                success=False,
                original_text=original_text,
                repaired_text=original_text,
                original_bbox=original_bbox,
                repaired_bbox=original_bbox,
                confidence_improvement=0.0,
                repair_methods=[],
                processing_time=time.time() - start_time
            )
    
    def _repair_text_content(self, text: str) -> Dict[str, Any]:
        """
        修复文本内容
        """
        repaired_text = text
        methods_used = []
        changed = False
        
        # 中文错误修正
        if self.repair_config['enable_chinese_correction']:
            for error, correction in self.text_repair_rules['chinese_corrections'].items():
                if error in repaired_text:
                    repaired_text = repaired_text.replace(error, correction)
                    methods_used.append(f'chinese_correction_{error}')
                    changed = True
        
        # 数字错误修正
        if self.repair_config['enable_number_correction']:
            # 修正数字中的字母
            number_pattern = r'\b\d*[a-zA-Z]+\d*\b'
            matches = re.findall(number_pattern, repaired_text)
            for match in matches:
                corrected = match
                for error, correction in self.text_repair_rules['number_corrections'].items():
                    corrected = corrected.replace(error, correction)
                if corrected != match:
                    repaired_text = repaired_text.replace(match, corrected)
                    methods_used.append('number_correction')
                    changed = True
        
        # 单位修正
        for error, correction in self.text_repair_rules['unit_corrections'].items():
            # 使用正则表达式确保只替换独立的单位
            pattern = r'\b' + re.escape(error) + r'\b'
            if re.search(pattern, repaired_text):
                repaired_text = re.sub(pattern, correction, repaired_text)
                methods_used.append(f'unit_correction_{error}')
                changed = True
        
        # 去除多余空格和特殊字符
        cleaned_text = re.sub(r'\s+', ' ', repaired_text).strip()
        if cleaned_text != repaired_text:
            repaired_text = cleaned_text
            methods_used.append('text_cleaning')
            changed = True
        
        return {
            'text': repaired_text,
            'methods': methods_used,
            'changed': changed
        }
    
    def _repair_coordinates(self, image: np.ndarray, bbox: List[float], text: str) -> Dict[str, Any]:
        """
        修复坐标
        """
        if len(bbox) < 4:
            return {'bbox': bbox, 'methods': [], 'changed': False, 'confidence_boost': 0.0}
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        
        repaired_bbox = [x1, y1, x2, y2]
        methods_used = []
        changed = False
        confidence_boost = 0.0
        
        # 1. 边界检查和修正
        original_bbox = repaired_bbox.copy()
        repaired_bbox[0] = max(0, min(repaired_bbox[0], w-1))  # x1
        repaired_bbox[1] = max(0, min(repaired_bbox[1], h-1))  # y1
        repaired_bbox[2] = max(repaired_bbox[0]+1, min(repaired_bbox[2], w))  # x2
        repaired_bbox[3] = max(repaired_bbox[1]+1, min(repaired_bbox[3], h))  # y2
        
        if repaired_bbox != original_bbox:
            methods_used.append('boundary_correction')
            changed = True
            confidence_boost += 0.05
        
        # 2. 扩展边界框（为了更好的识别）
        expansion = self.repair_config['coordinate_expansion']
        expanded_bbox = [
            max(0, repaired_bbox[0] - expansion),
            max(0, repaired_bbox[1] - expansion),
            min(w, repaired_bbox[2] + expansion),
            min(h, repaired_bbox[3] + expansion)
        ]
        
        # 检查扩展是否有效
        if expanded_bbox != repaired_bbox:
            # 提取扩展区域进行验证
            expanded_region = image[int(expanded_bbox[1]):int(expanded_bbox[3]), 
                                 int(expanded_bbox[0]):int(expanded_bbox[2])]
            
            if expanded_region.size > 0:
                # 简单验证：检查扩展区域是否包含更多文本信息
                if self._validate_expanded_region(expanded_region, text):
                    repaired_bbox = expanded_bbox
                    methods_used.append('bbox_expansion')
                    changed = True
                    confidence_boost += 0.1
        
        # 3. 使用轮廓检测精细化边界框
        try:
            refined_bbox = self._refine_bbox_with_contours(image, repaired_bbox)
            if refined_bbox and refined_bbox != repaired_bbox:
                repaired_bbox = refined_bbox
                methods_used.append('contour_refinement')
                changed = True
                confidence_boost += 0.15
        except Exception as e:
            self.logger.debug(f"轮廓精细化失败: {e}")
        
        return {
            'bbox': repaired_bbox,
            'methods': methods_used,
            'changed': changed,
            'confidence_boost': confidence_boost
        }
    
    def _validate_expanded_region(self, region: np.ndarray, expected_text: str) -> bool:
        """
        验证扩展区域是否有效
        """
        try:
            # 简单的验证：检查区域大小和内容
            if region.shape[0] < 10 or region.shape[1] < 10:
                return False
            
            # 检查区域是否包含足够的文本信息
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # 计算文本像素比例
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_pixels = np.sum(binary == 0)  # 假设文本是黑色
            total_pixels = binary.size
            
            text_ratio = text_pixels / total_pixels
            
            # 如果文本像素比例在合理范围内，认为扩展有效
            return 0.05 <= text_ratio <= 0.8
            
        except:
            return False
    
    def _refine_bbox_with_contours(self, image: np.ndarray, bbox: List[float]) -> Optional[List[float]]:
        """
        使用轮廓检测精细化边界框
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # 提取区域
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 合并所有轮廓的边界框
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # 转换回原图坐标
            refined_bbox = [
                x1 + x,
                y1 + y,
                x1 + x + w,
                y1 + y + h
            ]
            
            return refined_bbox
            
        except Exception as e:
            self.logger.debug(f"轮廓精细化失败: {e}")
            return None
    
    def _rerecognize_region(self, image: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
        """
        重新识别区域
        """
        try:
            if len(bbox) < 4:
                return {'success': False, 'text': '', 'confidence_boost': 0.0}
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # 提取区域
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                return {'success': False, 'text': '', 'confidence_boost': 0.0}
            
            # 使用中文OCR优化器重新识别
            result = self.chinese_optimizer.recognize_text_multi_scale(region)
            
            if result and result.get('text', '').strip():
                return {
                    'success': True,
                    'text': result['text'].strip(),
                    'confidence_boost': 0.2
                }
            
            # 如果中文优化器失败，尝试传统OCR
            try:
                # 增强图像
                enhanced_region = self.image_processor.enhance_for_ocr(region, 'text')
                
                # 使用Tesseract
                custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿克毫千焦卡路里蛋白质脂肪碳水化合物营养成分含量参考值项目能量'
                text = pytesseract.image_to_string(enhanced_region, lang='chi_sim+eng', config=custom_config)
                
                if text.strip():
                    return {
                        'success': True,
                        'text': text.strip(),
                        'confidence_boost': 0.15
                    }
            except:
                pass
            
            return {'success': False, 'text': '', 'confidence_boost': 0.0}
            
        except Exception as e:
            self.logger.debug(f"重新识别失败: {e}")
            return {'success': False, 'text': '', 'confidence_boost': 0.0}
    
    def repair_image_and_extract(self, image_path: str) -> Dict[str, Any]:
        """
        修复图像并提取信息
        
        Args:
            image_path: 图像路径
            
        Returns:
            修复后的提取结果
        """
        try:
            self.logger.info(f"开始修复和提取: {image_path}")
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 首次提取
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            original_result = self.extractor.extract_comprehensive_info(image, detection_results)
            
            # 修复OCR结果
            repaired_result = self.repair_ocr_result(image, original_result)
            
            # 添加比较信息
            comparison = self._compare_extraction_results(original_result, repaired_result)
            
            return {
                'image_path': image_path,
                'original_result': original_result,
                'repaired_result': repaired_result,
                'comparison': comparison,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"修复和提取失败: {e}")
            return {'error': str(e), 'image_path': image_path}
    
    def _compare_extraction_results(self, original: Dict, repaired: Dict) -> Dict[str, Any]:
        """
        比较提取结果
        """
        original_regions = original.get('text_regions', [])
        repaired_regions = repaired.get('text_regions', [])
        
        # 计算文本长度变化
        original_text_length = sum(len(r.get('text', '')) for r in original_regions)
        repaired_text_length = sum(len(r.get('text', '')) for r in repaired_regions)
        
        # 计算平均置信度变化
        original_confidences = [r.get('confidence', 0) for r in original_regions]
        repaired_confidences = [r.get('confidence', 0) for r in repaired_regions]
        
        original_avg_conf = np.mean(original_confidences) if original_confidences else 0
        repaired_avg_conf = np.mean(repaired_confidences) if repaired_confidences else 0
        
        # 统计修复的区域数
        repaired_count = sum(1 for r in repaired_regions if 'repair_info' in r)
        
        return {
            'text_length_change': repaired_text_length - original_text_length,
            'confidence_improvement': repaired_avg_conf - original_avg_conf,
            'regions_repaired': repaired_count,
            'total_regions': len(original_regions),
            'repair_rate': repaired_count / len(original_regions) if original_regions else 0,
            'improvement_percentage': ((repaired_avg_conf - original_avg_conf) / max(original_avg_conf, 0.1)) * 100
        }
    
    def batch_repair(self, input_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """
        批量修复图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（可选）
            
        Returns:
            批量修复结果
        """
        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise ValueError(f"输入目录不存在: {input_dir}")
            
            # 查找图像文件
            image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
            
            if not image_files:
                raise ValueError(f"在目录 {input_dir} 中未找到图像文件")
            
            self.logger.info(f"找到 {len(image_files)} 个图像文件")
            
            # 批量处理
            results = []
            for image_file in image_files:
                result = self.repair_image_and_extract(str(image_file))
                results.append(result)
            
            # 计算统计信息
            stats = self._calculate_batch_statistics(results)
            
            # 保存结果
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # 保存详细结果
                results_file = output_path / 'batch_repair_results.json'
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'statistics': stats,
                        'detailed_results': results
                    }, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"批量修复结果已保存到: {results_file}")
            
            return {
                'statistics': stats,
                'processed_files': len(image_files),
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"批量修复失败: {e}")
            return {'error': str(e)}
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算批量处理统计信息
        """
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {'error': '没有成功的处理结果'}
        
        # 收集比较数据
        comparisons = [r.get('comparison', {}) for r in successful_results]
        
        # 计算平均改进
        avg_confidence_improvement = np.mean([c.get('confidence_improvement', 0) for c in comparisons])
        avg_text_length_change = np.mean([c.get('text_length_change', 0) for c in comparisons])
        total_regions_repaired = sum(c.get('regions_repaired', 0) for c in comparisons)
        total_regions = sum(c.get('total_regions', 0) for c in comparisons)
        
        # 计算修复率
        overall_repair_rate = total_regions_repaired / total_regions if total_regions > 0 else 0
        
        return {
            'total_images_processed': len(successful_results),
            'average_confidence_improvement': avg_confidence_improvement,
            'average_text_length_change': avg_text_length_change,
            'total_regions_repaired': total_regions_repaired,
            'total_regions': total_regions,
            'overall_repair_rate': overall_repair_rate,
            'success_rate': len(successful_results) / len(results) if results else 0
        }

def main():
    """
    主函数
    """
    repair_tool = OCRRepairTool()
    
    # 测试单张图像
    test_image_path = "test_images/nutrition_label.jpg"
    if os.path.exists(test_image_path):
        print(f"修复单张图像: {test_image_path}")
        result = repair_tool.repair_image_and_extract(test_image_path)
        
        if 'comparison' in result:
            comparison = result['comparison']
            print(f"修复效果:")
            print(f"  置信度改进: {comparison['confidence_improvement']:.3f}")
            print(f"  修复区域数: {comparison['regions_repaired']}/{comparison['total_regions']}")
            print(f"  修复率: {comparison['repair_rate']:.2%}")
    
    # 测试批量修复
    test_dir = "test_images"
    if os.path.exists(test_dir):
        print(f"\n批量修复目录: {test_dir}")
        batch_result = repair_tool.batch_repair(test_dir, "repair_results")
        
        if 'statistics' in batch_result:
            stats = batch_result['statistics']
            print(f"批量修复统计:")
            print(f"  处理图像数: {stats['total_images_processed']}")
            print(f"  平均置信度改进: {stats['average_confidence_improvement']:.3f}")
            print(f"  总体修复率: {stats['overall_repair_rate']:.2%}")
    
    print("\nOCR修复工具测试完成！")

if __name__ == "__main__":
    main()