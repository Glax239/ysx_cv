#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强OCR识别测试脚本
测试改进后的OCR识别准确率和坐标定位精度
"""

import cv2
import numpy as np
import os
import sys
import logging
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.simple_information_extractor import SimpleInformationExtractor
from core.enhanced_image_processor import EnhancedImageProcessor
from core.chinese_ocr_optimizer import ChineseOCROptimizer

class EnhancedOCRTester:
    """
    增强OCR识别测试器
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
        
        # 测试结果
        self.test_results = []
    
    def test_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        测试单张图像的OCR识别效果
        
        Args:
            image_path: 图像路径
            
        Returns:
            测试结果字典
        """
        try:
            self.logger.info(f"开始测试图像: {image_path}")
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 1. 传统OCR识别
            traditional_result = self._test_traditional_ocr(image)
            
            # 2. 增强OCR识别
            enhanced_result = self._test_enhanced_ocr(image)
            
            # 3. 对比分析
            comparison = self._compare_results(traditional_result, enhanced_result)
            
            # 计算总耗时
            total_time = time.time() - start_time
            
            # 构建测试结果
            test_result = {
                'image_path': image_path,
                'image_size': image.shape,
                'traditional_ocr': traditional_result,
                'enhanced_ocr': enhanced_result,
                'comparison': comparison,
                'processing_time': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.test_results.append(test_result)
            self.logger.info(f"图像测试完成，耗时: {total_time:.2f}秒")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"测试图像失败: {e}")
            return {'error': str(e), 'image_path': image_path}
    
    def _test_traditional_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        测试传统OCR方法
        """
        try:
            start_time = time.time()
            
            # 使用简单的文本提取
            result = self.extractor.extract_text_simple(image)
            
            processing_time = time.time() - start_time
            
            return {
                'method': 'traditional',
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0),
                'processing_time': processing_time,
                'character_count': len(result.get('text', '')),
                'chinese_char_count': len([c for c in result.get('text', '') if '\u4e00' <= c <= '\u9fff'])
            }
            
        except Exception as e:
            self.logger.error(f"传统OCR测试失败: {e}")
            return {'method': 'traditional', 'error': str(e)}
    
    def _test_enhanced_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        测试增强OCR方法
        """
        try:
            start_time = time.time()
            
            # 使用增强的文本提取
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            result = self.extractor.extract_comprehensive_info(image, detection_results)
            
            processing_time = time.time() - start_time
            
            # 提取文本信息
            all_text = []
            all_confidences = []
            precise_bboxes = []
            
            if 'text_regions' in result:
                for region in result['text_regions']:
                    if 'text' in region and region['text'].strip():
                        all_text.append(region['text'])
                        all_confidences.append(region.get('confidence', 0))
                        if 'precise_bbox' in region:
                            precise_bboxes.append(region['precise_bbox'])
            
            combined_text = ' '.join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            return {
                'method': 'enhanced',
                'text': combined_text,
                'confidence': avg_confidence,
                'processing_time': processing_time,
                'character_count': len(combined_text),
                'chinese_char_count': len([c for c in combined_text if '\u4e00' <= c <= '\u9fff']),
                'region_count': len(all_text),
                'precise_bboxes': precise_bboxes,
                'detailed_regions': result.get('text_regions', [])
            }
            
        except Exception as e:
            self.logger.error(f"增强OCR测试失败: {e}")
            return {'method': 'enhanced', 'error': str(e)}
    
    def _compare_results(self, traditional: Dict, enhanced: Dict) -> Dict[str, Any]:
        """
        对比传统OCR和增强OCR的结果
        """
        comparison = {
            'text_length_improvement': 0,
            'chinese_char_improvement': 0,
            'confidence_improvement': 0,
            'processing_time_ratio': 0,
            'accuracy_score': 0
        }
        
        try:
            # 文本长度改进
            trad_len = traditional.get('character_count', 0)
            enh_len = enhanced.get('character_count', 0)
            if trad_len > 0:
                comparison['text_length_improvement'] = (enh_len - trad_len) / trad_len * 100
            
            # 中文字符改进
            trad_chinese = traditional.get('chinese_char_count', 0)
            enh_chinese = enhanced.get('chinese_char_count', 0)
            if trad_chinese > 0:
                comparison['chinese_char_improvement'] = (enh_chinese - trad_chinese) / trad_chinese * 100
            
            # 置信度改进
            trad_conf = traditional.get('confidence', 0)
            enh_conf = enhanced.get('confidence', 0)
            comparison['confidence_improvement'] = (enh_conf - trad_conf) * 100
            
            # 处理时间比较
            trad_time = traditional.get('processing_time', 1)
            enh_time = enhanced.get('processing_time', 1)
            comparison['processing_time_ratio'] = enh_time / trad_time
            
            # 综合准确率评分
            accuracy_factors = [
                min(100, max(0, comparison['text_length_improvement'])) * 0.3,
                min(100, max(0, comparison['chinese_char_improvement'])) * 0.4,
                min(100, max(0, comparison['confidence_improvement'])) * 0.3
            ]
            comparison['accuracy_score'] = sum(accuracy_factors)
            
            # 判断改进效果
            if comparison['accuracy_score'] > 20:
                comparison['improvement_level'] = '显著改进'
            elif comparison['accuracy_score'] > 10:
                comparison['improvement_level'] = '中等改进'
            elif comparison['accuracy_score'] > 0:
                comparison['improvement_level'] = '轻微改进'
            else:
                comparison['improvement_level'] = '无明显改进'
            
        except Exception as e:
            self.logger.error(f"结果对比失败: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def test_image_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        测试目录中的所有图像
        
        Args:
            directory_path: 图像目录路径
            
        Returns:
            所有测试结果列表
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"目录不存在: {directory_path}")
            
            # 支持的图像格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            # 查找所有图像文件
            image_files = []
            for ext in image_extensions:
                image_files.extend(directory.glob(f'*{ext}'))
                image_files.extend(directory.glob(f'*{ext.upper()}'))
            
            self.logger.info(f"找到 {len(image_files)} 个图像文件")
            
            # 测试每个图像
            results = []
            for i, image_file in enumerate(image_files, 1):
                self.logger.info(f"测试进度: {i}/{len(image_files)}")
                result = self.test_single_image(str(image_file))
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量测试失败: {e}")
            return []
    
    def generate_report(self, output_path: str = 'enhanced_ocr_test_report.json'):
        """
        生成测试报告
        
        Args:
            output_path: 报告输出路径
        """
        try:
            if not self.test_results:
                self.logger.warning("没有测试结果可生成报告")
                return
            
            # 计算统计信息
            stats = self._calculate_statistics()
            
            # 构建报告
            report = {
                'test_summary': {
                    'total_images': len(self.test_results),
                    'successful_tests': len([r for r in self.test_results if 'error' not in r]),
                    'failed_tests': len([r for r in self.test_results if 'error' in r]),
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'performance_statistics': stats,
                'detailed_results': self.test_results
            }
            
            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"测试报告已保存到: {output_path}")
            
            # 打印摘要
            self._print_summary(stats)
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        计算测试统计信息
        """
        successful_results = [r for r in self.test_results if 'error' not in r]
        
        if not successful_results:
            return {'error': '没有成功的测试结果'}
        
        # 提取比较数据
        improvements = [r['comparison'] for r in successful_results if 'comparison' in r]
        
        stats = {
            'average_text_length_improvement': 0,
            'average_chinese_char_improvement': 0,
            'average_confidence_improvement': 0,
            'average_processing_time_ratio': 0,
            'average_accuracy_score': 0,
            'improvement_distribution': {'显著改进': 0, '中等改进': 0, '轻微改进': 0, '无明显改进': 0}
        }
        
        if improvements:
            stats['average_text_length_improvement'] = sum(i.get('text_length_improvement', 0) for i in improvements) / len(improvements)
            stats['average_chinese_char_improvement'] = sum(i.get('chinese_char_improvement', 0) for i in improvements) / len(improvements)
            stats['average_confidence_improvement'] = sum(i.get('confidence_improvement', 0) for i in improvements) / len(improvements)
            stats['average_processing_time_ratio'] = sum(i.get('processing_time_ratio', 1) for i in improvements) / len(improvements)
            stats['average_accuracy_score'] = sum(i.get('accuracy_score', 0) for i in improvements) / len(improvements)
            
            # 改进程度分布
            for improvement in improvements:
                level = improvement.get('improvement_level', '无明显改进')
                if level in stats['improvement_distribution']:
                    stats['improvement_distribution'][level] += 1
        
        return stats
    
    def _print_summary(self, stats: Dict[str, Any]):
        """
        打印测试摘要
        """
        print("\n" + "="*60)
        print("增强OCR识别测试报告摘要")
        print("="*60)
        
        if 'error' in stats:
            print(f"错误: {stats['error']}")
            return
        
        print(f"平均文本长度改进: {stats['average_text_length_improvement']:.2f}%")
        print(f"平均中文字符改进: {stats['average_chinese_char_improvement']:.2f}%")
        print(f"平均置信度改进: {stats['average_confidence_improvement']:.2f}%")
        print(f"平均处理时间比例: {stats['average_processing_time_ratio']:.2f}x")
        print(f"平均准确率评分: {stats['average_accuracy_score']:.2f}")
        
        print("\n改进程度分布:")
        for level, count in stats['improvement_distribution'].items():
            print(f"  {level}: {count} 个图像")
        
        print("="*60)

def main():
    """
    主函数
    """
    tester = EnhancedOCRTester()
    
    # 测试单张图像（如果存在）
    test_image_path = "test_images/nutrition_label.jpg"
    if os.path.exists(test_image_path):
        print(f"测试单张图像: {test_image_path}")
        result = tester.test_single_image(test_image_path)
        print(f"测试结果: {result}")
    
    # 测试图像目录（如果存在）
    test_dir = "test_images"
    if os.path.exists(test_dir):
        print(f"\n测试图像目录: {test_dir}")
        results = tester.test_image_directory(test_dir)
        print(f"批量测试完成，共测试 {len(results)} 个图像")
    
    # 生成报告
    tester.generate_report()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()