#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标精度验证工具
专门测试OCR识别的坐标定位准确性，避免识别出来但无法解析的情况
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.simple_information_extractor import SimpleInformationExtractor
from core.enhanced_image_processor import EnhancedImageProcessor

class CoordinateAccuracyValidator:
    """
    坐标精度验证器
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
        
        # 验证结果
        self.validation_results = []
        
        # 设置中文字体（用于可视化）
        try:
            self.chinese_font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
        except:
            self.chinese_font = None
            self.logger.warning("未找到中文字体，可视化可能显示异常")
    
    def validate_image_coordinates(self, image_path: str, save_visualization: bool = True) -> Dict[str, Any]:
        """
        验证单张图像的坐标精度
        
        Args:
            image_path: 图像路径
            save_visualization: 是否保存可视化结果
            
        Returns:
            验证结果字典
        """
        try:
            self.logger.info(f"开始验证图像坐标: {image_path}")
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 转换为RGB（用于matplotlib显示）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 提取文本区域
            start_time = time.time()
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            extraction_result = self.extractor.extract_comprehensive_info(image, detection_results)
            processing_time = time.time() - start_time
            
            # 验证坐标精度
            coordinate_validation = self._validate_coordinates(image, extraction_result)
            
            # 分析文本解析能力
            parsing_analysis = self._analyze_text_parsing(extraction_result)
            
            # 构建验证结果
            validation_result = {
                'image_path': image_path,
                'image_size': image.shape,
                'processing_time': processing_time,
                'coordinate_validation': coordinate_validation,
                'parsing_analysis': parsing_analysis,
                'extraction_result': extraction_result,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存可视化结果
            if save_visualization:
                viz_path = self._create_visualization(image_rgb, validation_result)
                validation_result['visualization_path'] = viz_path
            
            self.validation_results.append(validation_result)
            self.logger.info(f"坐标验证完成，耗时: {processing_time:.2f}秒")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"坐标验证失败: {e}")
            return {'error': str(e), 'image_path': image_path}
    
    def _validate_coordinates(self, image: np.ndarray, extraction_result: Dict) -> Dict[str, Any]:
        """
        验证坐标精度
        """
        validation = {
            'total_regions': 0,
            'valid_coordinates': 0,
            'coordinate_accuracy': 0,
            'bbox_coverage': 0,
            'precise_bbox_improvement': 0,
            'coordinate_issues': []
        }
        
        try:
            text_regions = extraction_result.get('text_regions', [])
            validation['total_regions'] = len(text_regions)
            
            if not text_regions:
                validation['coordinate_issues'].append('未检测到任何文本区域')
                return validation
            
            h, w = image.shape[:2]
            image_area = h * w
            total_bbox_area = 0
            valid_coords = 0
            precise_improvements = []
            
            for i, region in enumerate(text_regions):
                # 检查基本坐标
                bbox = region.get('bbox', [])
                precise_bbox = region.get('precise_bbox', [])
                
                if len(bbox) >= 4:
                    # 验证坐标范围
                    x1, y1, x2, y2 = bbox[:4]
                    if self._is_valid_bbox(x1, y1, x2, y2, w, h):
                        valid_coords += 1
                        
                        # 计算边界框面积
                        bbox_area = (x2 - x1) * (y2 - y1)
                        total_bbox_area += bbox_area
                        
                        # 比较精确边界框
                        if len(precise_bbox) >= 4:
                            px1, py1, px2, py2 = precise_bbox[:4]
                            if self._is_valid_bbox(px1, py1, px2, py2, w, h):
                                # 计算精确度改进
                                precise_area = (px2 - px1) * (py2 - py1)
                                if bbox_area > 0:
                                    improvement = abs(precise_area - bbox_area) / bbox_area
                                    precise_improvements.append(improvement)
                    else:
                        validation['coordinate_issues'].append(f'区域 {i+1} 坐标超出图像范围: {bbox}')
                else:
                    validation['coordinate_issues'].append(f'区域 {i+1} 坐标格式错误: {bbox}')
            
            # 计算指标
            validation['valid_coordinates'] = valid_coords
            validation['coordinate_accuracy'] = (valid_coords / len(text_regions)) * 100 if text_regions else 0
            validation['bbox_coverage'] = (total_bbox_area / image_area) * 100 if image_area > 0 else 0
            validation['precise_bbox_improvement'] = np.mean(precise_improvements) * 100 if precise_improvements else 0
            
        except Exception as e:
            validation['coordinate_issues'].append(f'坐标验证异常: {str(e)}')
        
        return validation
    
    def _analyze_text_parsing(self, extraction_result: Dict) -> Dict[str, Any]:
        """
        分析文本解析能力
        """
        analysis = {
            'total_text_length': 0,
            'chinese_char_count': 0,
            'english_char_count': 0,
            'number_count': 0,
            'special_char_count': 0,
            'empty_regions': 0,
            'low_confidence_regions': 0,
            'parsing_success_rate': 0,
            'average_confidence': 0,
            'text_distribution': {},
            'parsing_issues': []
        }
        
        try:
            text_regions = extraction_result.get('text_regions', [])
            
            if not text_regions:
                analysis['parsing_issues'].append('没有文本区域可分析')
                return analysis
            
            confidences = []
            all_text = ""
            
            for i, region in enumerate(text_regions):
                text = region.get('text', '').strip()
                confidence = region.get('confidence', 0)
                
                if not text:
                    analysis['empty_regions'] += 1
                    analysis['parsing_issues'].append(f'区域 {i+1} 识别为空')
                    continue
                
                if confidence < 0.5:
                    analysis['low_confidence_regions'] += 1
                    analysis['parsing_issues'].append(f'区域 {i+1} 置信度过低: {confidence:.3f}')
                
                confidences.append(confidence)
                all_text += text + " "
                
                # 分析字符类型
                for char in text:
                    if '\u4e00' <= char <= '\u9fff':  # 中文字符
                        analysis['chinese_char_count'] += 1
                    elif char.isalpha():  # 英文字符
                        analysis['english_char_count'] += 1
                    elif char.isdigit():  # 数字
                        analysis['number_count'] += 1
                    else:  # 特殊字符
                        analysis['special_char_count'] += 1
            
            # 计算统计信息
            analysis['total_text_length'] = len(all_text.strip())
            analysis['parsing_success_rate'] = ((len(text_regions) - analysis['empty_regions']) / len(text_regions)) * 100 if text_regions else 0
            analysis['average_confidence'] = np.mean(confidences) if confidences else 0
            
            # 文本分布
            total_chars = analysis['chinese_char_count'] + analysis['english_char_count'] + analysis['number_count'] + analysis['special_char_count']
            if total_chars > 0:
                analysis['text_distribution'] = {
                    'chinese_percentage': (analysis['chinese_char_count'] / total_chars) * 100,
                    'english_percentage': (analysis['english_char_count'] / total_chars) * 100,
                    'number_percentage': (analysis['number_count'] / total_chars) * 100,
                    'special_percentage': (analysis['special_char_count'] / total_chars) * 100
                }
            
            # 检查营养成分表特征
            nutrition_keywords = ['能量', '蛋白质', '脂肪', '碳水化合物', '钠', '营养', '成分']
            found_keywords = [kw for kw in nutrition_keywords if kw in all_text]
            if found_keywords:
                analysis['nutrition_keywords_found'] = found_keywords
                analysis['is_nutrition_table'] = len(found_keywords) >= 3
            
        except Exception as e:
            analysis['parsing_issues'].append(f'文本分析异常: {str(e)}')
        
        return analysis
    
    def _is_valid_bbox(self, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> bool:
        """
        验证边界框是否有效
        """
        try:
            # 检查坐标顺序
            if x1 >= x2 or y1 >= y2:
                return False
            
            # 检查坐标范围
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                return False
            
            # 检查最小尺寸
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                return False
            
            return True
            
        except:
            return False
    
    def _create_visualization(self, image_rgb: np.ndarray, validation_result: Dict) -> str:
        """
        创建坐标验证可视化图像
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # 左图：原图 + YOLO检测框
            ax1.imshow(image_rgb)
            ax1.set_title('YOLO检测边界框', fontproperties=self.chinese_font, fontsize=14)
            ax1.axis('off')
            
            # 右图：原图 + 精确OCR边界框
            ax2.imshow(image_rgb)
            ax2.set_title('OCR精确边界框', fontproperties=self.chinese_font, fontsize=14)
            ax2.axis('off')
            
            # 绘制边界框
            text_regions = validation_result.get('extraction_result', {}).get('text_regions', [])
            
            for i, region in enumerate(text_regions):
                text = region.get('text', '').strip()
                bbox = region.get('bbox', [])
                precise_bbox = region.get('precise_bbox', [])
                confidence = region.get('confidence', 0)
                
                # 绘制YOLO检测框
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    rect1 = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            linewidth=2, edgecolor='red', facecolor='none')
                    ax1.add_patch(rect1)
                    
                    # 添加文本标签
                    label1 = f'{i+1}: {text[:10]}...' if len(text) > 10 else f'{i+1}: {text}'
                    ax1.text(x1, y1-5, label1, fontproperties=self.chinese_font, 
                           fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                # 绘制精确OCR边界框
                if len(precise_bbox) >= 4:
                    px1, py1, px2, py2 = precise_bbox[:4]
                    rect2 = patches.Rectangle((px1, py1), px2-px1, py2-py1, 
                                            linewidth=2, edgecolor='blue', facecolor='none')
                    ax2.add_patch(rect2)
                    
                    # 添加文本标签
                    label2 = f'{i+1}: {text[:10]}... (conf: {confidence:.2f})' if len(text) > 10 else f'{i+1}: {text} (conf: {confidence:.2f})'
                    ax2.text(px1, py1-5, label2, fontproperties=self.chinese_font, 
                           fontsize=8, color='blue', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 添加统计信息
            coord_val = validation_result.get('coordinate_validation', {})
            parse_anal = validation_result.get('parsing_analysis', {})
            
            stats_text = f"""坐标验证统计:
总区域数: {coord_val.get('total_regions', 0)}
有效坐标: {coord_val.get('valid_coordinates', 0)}
坐标准确率: {coord_val.get('coordinate_accuracy', 0):.1f}%
边界框覆盖率: {coord_val.get('bbox_coverage', 0):.1f}%

文本解析统计:
总文本长度: {parse_anal.get('total_text_length', 0)}
中文字符: {parse_anal.get('chinese_char_count', 0)}
解析成功率: {parse_anal.get('parsing_success_rate', 0):.1f}%
平均置信度: {parse_anal.get('average_confidence', 0):.3f}"""
            
            fig.text(0.02, 0.02, stats_text, fontproperties=self.chinese_font, 
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图像
            image_name = Path(validation_result['image_path']).stem
            viz_path = f'coordinate_validation_{image_name}_{int(time.time())}.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"可视化结果已保存: {viz_path}")
            return viz_path
            
        except Exception as e:
            self.logger.error(f"创建可视化失败: {e}")
            return ""
    
    def generate_validation_report(self, output_path: str = 'coordinate_validation_report.json'):
        """
        生成坐标验证报告
        
        Args:
            output_path: 报告输出路径
        """
        try:
            if not self.validation_results:
                self.logger.warning("没有验证结果可生成报告")
                return
            
            # 计算统计信息
            stats = self._calculate_validation_statistics()
            
            # 构建报告
            report = {
                'validation_summary': {
                    'total_images': len(self.validation_results),
                    'successful_validations': len([r for r in self.validation_results if 'error' not in r]),
                    'failed_validations': len([r for r in self.validation_results if 'error' in r]),
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'coordinate_accuracy_statistics': stats,
                'detailed_results': self.validation_results
            }
            
            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"验证报告已保存到: {output_path}")
            
            # 打印摘要
            self._print_validation_summary(stats)
            
        except Exception as e:
            self.logger.error(f"生成验证报告失败: {e}")
    
    def _calculate_validation_statistics(self) -> Dict[str, Any]:
        """
        计算验证统计信息
        """
        successful_results = [r for r in self.validation_results if 'error' not in r]
        
        if not successful_results:
            return {'error': '没有成功的验证结果'}
        
        stats = {
            'average_coordinate_accuracy': 0,
            'average_bbox_coverage': 0,
            'average_parsing_success_rate': 0,
            'average_confidence': 0,
            'total_coordinate_issues': 0,
            'total_parsing_issues': 0,
            'chinese_text_percentage': 0,
            'nutrition_table_detection_rate': 0
        }
        
        coord_accuracies = []
        bbox_coverages = []
        parsing_rates = []
        confidences = []
        coord_issues = 0
        parsing_issues = 0
        chinese_percentages = []
        nutrition_detections = 0
        
        for result in successful_results:
            coord_val = result.get('coordinate_validation', {})
            parse_anal = result.get('parsing_analysis', {})
            
            coord_accuracies.append(coord_val.get('coordinate_accuracy', 0))
            bbox_coverages.append(coord_val.get('bbox_coverage', 0))
            parsing_rates.append(parse_anal.get('parsing_success_rate', 0))
            confidences.append(parse_anal.get('average_confidence', 0))
            
            coord_issues += len(coord_val.get('coordinate_issues', []))
            parsing_issues += len(parse_anal.get('parsing_issues', []))
            
            text_dist = parse_anal.get('text_distribution', {})
            chinese_percentages.append(text_dist.get('chinese_percentage', 0))
            
            if parse_anal.get('is_nutrition_table', False):
                nutrition_detections += 1
        
        # 计算平均值
        if successful_results:
            stats['average_coordinate_accuracy'] = np.mean(coord_accuracies)
            stats['average_bbox_coverage'] = np.mean(bbox_coverages)
            stats['average_parsing_success_rate'] = np.mean(parsing_rates)
            stats['average_confidence'] = np.mean(confidences)
            stats['total_coordinate_issues'] = coord_issues
            stats['total_parsing_issues'] = parsing_issues
            stats['chinese_text_percentage'] = np.mean(chinese_percentages)
            stats['nutrition_table_detection_rate'] = (nutrition_detections / len(successful_results)) * 100
        
        return stats
    
    def _print_validation_summary(self, stats: Dict[str, Any]):
        """
        打印验证摘要
        """
        print("\n" + "="*60)
        print("坐标精度验证报告摘要")
        print("="*60)
        
        if 'error' in stats:
            print(f"错误: {stats['error']}")
            return
        
        print(f"平均坐标准确率: {stats['average_coordinate_accuracy']:.2f}%")
        print(f"平均边界框覆盖率: {stats['average_bbox_coverage']:.2f}%")
        print(f"平均解析成功率: {stats['average_parsing_success_rate']:.2f}%")
        print(f"平均识别置信度: {stats['average_confidence']:.3f}")
        print(f"坐标问题总数: {stats['total_coordinate_issues']}")
        print(f"解析问题总数: {stats['total_parsing_issues']}")
        print(f"中文文本占比: {stats['chinese_text_percentage']:.2f}%")
        print(f"营养成分表检测率: {stats['nutrition_table_detection_rate']:.2f}%")
        
        print("="*60)

def main():
    """
    主函数
    """
    validator = CoordinateAccuracyValidator()
    
    # 测试单张图像（如果存在）
    test_image_path = "test_images/nutrition_label.jpg"
    if os.path.exists(test_image_path):
        print(f"验证单张图像坐标: {test_image_path}")
        result = validator.validate_image_coordinates(test_image_path)
        print(f"验证结果: {result}")
    
    # 测试图像目录（如果存在）
    test_dir = "test_images"
    if os.path.exists(test_dir):
        print(f"\n批量验证图像目录: {test_dir}")
        image_files = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
        for image_file in image_files:
            validator.validate_image_coordinates(str(image_file))
    
    # 生成报告
    validator.generate_validation_report()
    
    print("\n坐标验证完成！")

if __name__ == "__main__":
    main()