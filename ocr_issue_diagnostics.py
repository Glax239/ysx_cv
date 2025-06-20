#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR问题诊断和修复工具
专门诊断和修复OCR识别中的常见问题，提高识别准确率和坐标精度
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
from dataclasses import dataclass
import re

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.simple_information_extractor import SimpleInformationExtractor
from core.enhanced_image_processor import EnhancedImageProcessor
from core.chinese_ocr_optimizer import ChineseOCROptimizer

@dataclass
class OCRIssue:
    """OCR问题数据类"""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    suggested_fix: str
    region_id: Optional[int] = None
    confidence_impact: float = 0.0

class OCRIssueDiagnostics:
    """
    OCR问题诊断器
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
        
        # 诊断结果
        self.diagnostic_results = []
        
        # 问题检测规则
        self.issue_patterns = {
            'low_confidence': {
                'threshold': 0.3,
                'severity': 'high',
                'description': '识别置信度过低',
                'fix': '增强图像预处理，调整OCR参数'
            },
            'empty_text': {
                'severity': 'critical',
                'description': '识别结果为空',
                'fix': '检查图像质量，尝试不同的预处理方法'
            },
            'garbled_text': {
                'severity': 'high',
                'description': '识别结果包含乱码',
                'fix': '优化字符白名单，增强图像清晰度'
            },
            'coordinate_overflow': {
                'severity': 'medium',
                'description': '坐标超出图像范围',
                'fix': '修正坐标计算逻辑'
            },
            'small_region': {
                'threshold': 100,  # 像素面积
                'severity': 'medium',
                'description': '检测区域过小',
                'fix': '调整YOLO检测阈值，优化区域合并'
            },
            'overlapping_regions': {
                'threshold': 0.5,  # IoU阈值
                'severity': 'medium',
                'description': '区域重叠过多',
                'fix': '优化非极大值抑制参数'
            },
            'chinese_recognition_error': {
                'severity': 'high',
                'description': '中文识别错误',
                'fix': '使用中文OCR优化器，调整语言模型'
            },
            'number_recognition_error': {
                'severity': 'medium',
                'description': '数字识别错误',
                'fix': '使用数字专用OCR配置'
            }
        }
    
    def diagnose_image(self, image_path: str) -> Dict[str, Any]:
        """
        诊断单张图像的OCR问题
        
        Args:
            image_path: 图像路径
            
        Returns:
            诊断结果字典
        """
        try:
            self.logger.info(f"开始诊断图像: {image_path}")
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 执行OCR识别
            start_time = time.time()
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            extraction_result = self.extractor.extract_comprehensive_info(image, detection_results)
            processing_time = time.time() - start_time
            
            # 诊断问题
            issues = self._detect_issues(image, extraction_result)
            
            # 生成修复建议
            fixes = self._generate_fixes(issues, image, extraction_result)
            
            # 评估整体质量
            quality_score = self._calculate_quality_score(extraction_result, issues)
            
            # 构建诊断结果
            diagnostic_result = {
                'image_path': image_path,
                'image_size': image.shape,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'issues_detected': len(issues),
                'issues': [self._issue_to_dict(issue) for issue in issues],
                'fixes': fixes,
                'extraction_result': extraction_result,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.diagnostic_results.append(diagnostic_result)
            self.logger.info(f"诊断完成，发现 {len(issues)} 个问题，质量评分: {quality_score:.2f}")
            
            return diagnostic_result
            
        except Exception as e:
            self.logger.error(f"图像诊断失败: {e}")
            return {'error': str(e), 'image_path': image_path}
    
    def _detect_issues(self, image: np.ndarray, extraction_result: Dict) -> List[OCRIssue]:
        """
        检测OCR问题
        """
        issues = []
        text_regions = extraction_result.get('text_regions', [])
        h, w = image.shape[:2]
        
        # 检查是否没有检测到任何区域
        if not text_regions:
            issues.append(OCRIssue(
                issue_type='no_regions_detected',
                severity='critical',
                description='未检测到任何文本区域',
                suggested_fix='检查YOLO模型配置，降低检测阈值'
            ))
            return issues
        
        # 逐个检查文本区域
        for i, region in enumerate(text_regions):
            region_issues = self._check_region_issues(region, i, w, h)
            issues.extend(region_issues)
        
        # 检查区域间问题
        global_issues = self._check_global_issues(text_regions, w, h)
        issues.extend(global_issues)
        
        return issues
    
    def _check_region_issues(self, region: Dict, region_id: int, img_w: int, img_h: int) -> List[OCRIssue]:
        """
        检查单个区域的问题
        """
        issues = []
        
        text = region.get('text', '').strip()
        confidence = region.get('confidence', 0)
        bbox = region.get('bbox', [])
        
        # 检查空文本
        if not text:
            issues.append(OCRIssue(
                issue_type='empty_text',
                severity=self.issue_patterns['empty_text']['severity'],
                description=f"区域 {region_id+1} {self.issue_patterns['empty_text']['description']}",
                suggested_fix=self.issue_patterns['empty_text']['fix'],
                region_id=region_id,
                confidence_impact=1.0
            ))
        
        # 检查低置信度
        if confidence < self.issue_patterns['low_confidence']['threshold']:
            issues.append(OCRIssue(
                issue_type='low_confidence',
                severity=self.issue_patterns['low_confidence']['severity'],
                description=f"区域 {region_id+1} {self.issue_patterns['low_confidence']['description']} ({confidence:.3f})",
                suggested_fix=self.issue_patterns['low_confidence']['fix'],
                region_id=region_id,
                confidence_impact=self.issue_patterns['low_confidence']['threshold'] - confidence
            ))
        
        # 检查乱码
        if text and self._is_garbled_text(text):
            issues.append(OCRIssue(
                issue_type='garbled_text',
                severity=self.issue_patterns['garbled_text']['severity'],
                description=f"区域 {region_id+1} {self.issue_patterns['garbled_text']['description']}: '{text}'",
                suggested_fix=self.issue_patterns['garbled_text']['fix'],
                region_id=region_id,
                confidence_impact=0.5
            ))
        
        # 检查坐标问题
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            
            # 坐标超出范围
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                issues.append(OCRIssue(
                    issue_type='coordinate_overflow',
                    severity=self.issue_patterns['coordinate_overflow']['severity'],
                    description=f"区域 {region_id+1} {self.issue_patterns['coordinate_overflow']['description']}: ({x1}, {y1}, {x2}, {y2})",
                    suggested_fix=self.issue_patterns['coordinate_overflow']['fix'],
                    region_id=region_id,
                    confidence_impact=0.2
                ))
            
            # 区域过小
            area = (x2 - x1) * (y2 - y1)
            if area < self.issue_patterns['small_region']['threshold']:
                issues.append(OCRIssue(
                    issue_type='small_region',
                    severity=self.issue_patterns['small_region']['severity'],
                    description=f"区域 {region_id+1} {self.issue_patterns['small_region']['description']} (面积: {area})",
                    suggested_fix=self.issue_patterns['small_region']['fix'],
                    region_id=region_id,
                    confidence_impact=0.3
                ))
        
        # 检查中文识别错误
        if text and self._has_chinese_recognition_error(text):
            issues.append(OCRIssue(
                issue_type='chinese_recognition_error',
                severity=self.issue_patterns['chinese_recognition_error']['severity'],
                description=f"区域 {region_id+1} {self.issue_patterns['chinese_recognition_error']['description']}: '{text}'",
                suggested_fix=self.issue_patterns['chinese_recognition_error']['fix'],
                region_id=region_id,
                confidence_impact=0.4
            ))
        
        # 检查数字识别错误
        if text and self._has_number_recognition_error(text):
            issues.append(OCRIssue(
                issue_type='number_recognition_error',
                severity=self.issue_patterns['number_recognition_error']['severity'],
                description=f"区域 {region_id+1} {self.issue_patterns['number_recognition_error']['description']}: '{text}'",
                suggested_fix=self.issue_patterns['number_recognition_error']['fix'],
                region_id=region_id,
                confidence_impact=0.3
            ))
        
        return issues
    
    def _check_global_issues(self, text_regions: List[Dict], img_w: int, img_h: int) -> List[OCRIssue]:
        """
        检查全局问题
        """
        issues = []
        
        # 检查区域重叠
        overlapping_pairs = self._find_overlapping_regions(text_regions)
        if overlapping_pairs:
            issues.append(OCRIssue(
                issue_type='overlapping_regions',
                severity=self.issue_patterns['overlapping_regions']['severity'],
                description=f"{self.issue_patterns['overlapping_regions']['description']} ({len(overlapping_pairs)} 对)",
                suggested_fix=self.issue_patterns['overlapping_regions']['fix'],
                confidence_impact=0.2 * len(overlapping_pairs)
            ))
        
        return issues
    
    def _is_garbled_text(self, text: str) -> bool:
        """
        检查是否为乱码
        """
        # 检查特殊字符比例
        special_chars = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in '.,!?()[]{}"\'-+=*/%<>:;'))
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # 检查常见乱码模式
        garbled_patterns = [
            r'[\x00-\x1f\x7f-\x9f]',  # 控制字符
            r'[\ufffd]',  # 替换字符
            r'[\u0080-\u00ff]{3,}',  # 连续的扩展ASCII
        ]
        
        for pattern in garbled_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _has_chinese_recognition_error(self, text: str) -> bool:
        """
        检查中文识别错误
        """
        # 常见中文OCR错误模式
        error_patterns = [
            (r'能最', '能量'),
            (r'蛋白贡', '蛋白质'),
            (r'脂防', '脂肪'),
            (r'碳水化台物', '碳水化合物'),
            (r'营美', '营养'),
            (r'戍分', '成分'),
            (r'含童', '含量'),
            (r'参考直', '参考值')
        ]
        
        for error_pattern, _ in error_patterns:
            if re.search(error_pattern, text):
                return True
        
        return False
    
    def _has_number_recognition_error(self, text: str) -> bool:
        """
        检查数字识别错误
        """
        # 检查数字中的字母（常见OCR错误）
        number_with_letters = re.search(r'\d+[a-zA-Z]+\d*|[a-zA-Z]+\d+', text)
        if number_with_letters:
            return True
        
        # 检查不合理的数字格式
        weird_numbers = re.search(r'\d{10,}|\d+\.\d{5,}', text)
        if weird_numbers:
            return True
        
        return False
    
    def _find_overlapping_regions(self, text_regions: List[Dict]) -> List[Tuple[int, int]]:
        """
        查找重叠的区域
        """
        overlapping_pairs = []
        
        for i in range(len(text_regions)):
            for j in range(i + 1, len(text_regions)):
                bbox1 = text_regions[i].get('bbox', [])
                bbox2 = text_regions[j].get('bbox', [])
                
                if len(bbox1) >= 4 and len(bbox2) >= 4:
                    iou = self._calculate_iou(bbox1, bbox2)
                    if iou > self.issue_patterns['overlapping_regions']['threshold']:
                        overlapping_pairs.append((i, j))
        
        return overlapping_pairs
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个边界框的IoU
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
            x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
            
            # 计算交集
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # 计算并集
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except:
            return 0.0
    
    def _generate_fixes(self, issues: List[OCRIssue], image: np.ndarray, extraction_result: Dict) -> Dict[str, Any]:
        """
        生成修复建议
        """
        fixes = {
            'immediate_fixes': [],
            'parameter_adjustments': [],
            'preprocessing_improvements': [],
            'model_optimizations': []
        }
        
        # 根据问题类型生成修复建议
        issue_types = [issue.issue_type for issue in issues]
        
        if 'low_confidence' in issue_types:
            fixes['preprocessing_improvements'].append({
                'action': '增强图像预处理',
                'details': '使用更强的去噪和对比度增强',
                'priority': 'high'
            })
        
        if 'empty_text' in issue_types:
            fixes['parameter_adjustments'].append({
                'action': '调整OCR参数',
                'details': '尝试不同的PSM模式和OEM引擎',
                'priority': 'critical'
            })
        
        if 'chinese_recognition_error' in issue_types:
            fixes['model_optimizations'].append({
                'action': '启用中文OCR优化器',
                'details': '使用专门的中文识别配置和错误纠正',
                'priority': 'high'
            })
        
        if 'coordinate_overflow' in issue_types:
            fixes['immediate_fixes'].append({
                'action': '修正坐标计算',
                'details': '添加边界检查和坐标裁剪',
                'priority': 'medium'
            })
        
        if 'overlapping_regions' in issue_types:
            fixes['parameter_adjustments'].append({
                'action': '优化NMS参数',
                'details': '调整非极大值抑制的IoU阈值',
                'priority': 'medium'
            })
        
        return fixes
    
    def _calculate_quality_score(self, extraction_result: Dict, issues: List[OCRIssue]) -> float:
        """
        计算整体质量评分
        """
        base_score = 100.0
        
        # 根据问题严重程度扣分
        severity_penalties = {
            'critical': 25.0,
            'high': 15.0,
            'medium': 8.0,
            'low': 3.0
        }
        
        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 5.0)
            base_score -= penalty
        
        # 根据置信度影响进一步调整
        confidence_penalty = sum(issue.confidence_impact for issue in issues) * 10
        base_score -= confidence_penalty
        
        # 确保分数在0-100范围内
        return max(0.0, min(100.0, base_score))
    
    def _issue_to_dict(self, issue: OCRIssue) -> Dict[str, Any]:
        """
        将OCRIssue对象转换为字典
        """
        return {
            'issue_type': issue.issue_type,
            'severity': issue.severity,
            'description': issue.description,
            'suggested_fix': issue.suggested_fix,
            'region_id': issue.region_id,
            'confidence_impact': issue.confidence_impact
        }
    
    def apply_automatic_fixes(self, image_path: str) -> Dict[str, Any]:
        """
        自动应用修复措施
        
        Args:
            image_path: 图像路径
            
        Returns:
            修复结果
        """
        try:
            self.logger.info(f"开始自动修复: {image_path}")
            
            # 先诊断问题
            diagnostic_result = self.diagnose_image(image_path)
            
            if 'error' in diagnostic_result:
                return diagnostic_result
            
            # 加载图像
            image = cv2.imread(image_path)
            
            # 应用修复措施
            fixed_image = self._apply_image_fixes(image, diagnostic_result['issues'])
            
            # 重新识别
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            fixed_result = self.extractor.extract_comprehensive_info(fixed_image, detection_results)
            
            # 比较修复效果
            improvement = self._compare_results(
                diagnostic_result['extraction_result'],
                fixed_result
            )
            
            return {
                'original_diagnostic': diagnostic_result,
                'fixed_result': fixed_result,
                'improvement': improvement,
                'fixes_applied': True
            }
            
        except Exception as e:
            self.logger.error(f"自动修复失败: {e}")
            return {'error': str(e), 'image_path': image_path}
    
    def _apply_image_fixes(self, image: np.ndarray, issues: List[Dict]) -> np.ndarray:
        """
        应用图像修复
        """
        fixed_image = image.copy()
        
        # 检查是否需要增强预处理
        needs_enhancement = any(
            issue['issue_type'] in ['low_confidence', 'garbled_text', 'empty_text']
            for issue in issues
        )
        
        if needs_enhancement:
            # 应用增强的图像预处理
            fixed_image = self.image_processor.enhance_for_ocr(fixed_image, 'nutrition')
        
        return fixed_image
    
    def _compare_results(self, original: Dict, fixed: Dict) -> Dict[str, Any]:
        """
        比较修复前后的结果
        """
        original_regions = original.get('text_regions', [])
        fixed_regions = fixed.get('text_regions', [])
        
        # 计算改进指标
        original_text_length = sum(len(r.get('text', '')) for r in original_regions)
        fixed_text_length = sum(len(r.get('text', '')) for r in fixed_regions)
        
        original_avg_conf = np.mean([r.get('confidence', 0) for r in original_regions]) if original_regions else 0
        fixed_avg_conf = np.mean([r.get('confidence', 0) for r in fixed_regions]) if fixed_regions else 0
        
        return {
            'text_length_improvement': fixed_text_length - original_text_length,
            'confidence_improvement': fixed_avg_conf - original_avg_conf,
            'region_count_change': len(fixed_regions) - len(original_regions),
            'improvement_percentage': ((fixed_avg_conf - original_avg_conf) / max(original_avg_conf, 0.1)) * 100
        }
    
    def generate_diagnostic_report(self, output_path: str = 'ocr_diagnostic_report.json'):
        """
        生成诊断报告
        
        Args:
            output_path: 报告输出路径
        """
        try:
            if not self.diagnostic_results:
                self.logger.warning("没有诊断结果可生成报告")
                return
            
            # 计算统计信息
            stats = self._calculate_diagnostic_statistics()
            
            # 构建报告
            report = {
                'diagnostic_summary': {
                    'total_images': len(self.diagnostic_results),
                    'average_quality_score': stats['average_quality_score'],
                    'total_issues': stats['total_issues'],
                    'most_common_issues': stats['most_common_issues'],
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'issue_statistics': stats,
                'detailed_results': self.diagnostic_results
            }
            
            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"诊断报告已保存到: {output_path}")
            
            # 打印摘要
            self._print_diagnostic_summary(stats)
            
        except Exception as e:
            self.logger.error(f"生成诊断报告失败: {e}")
    
    def _calculate_diagnostic_statistics(self) -> Dict[str, Any]:
        """
        计算诊断统计信息
        """
        successful_results = [r for r in self.diagnostic_results if 'error' not in r]
        
        if not successful_results:
            return {'error': '没有成功的诊断结果'}
        
        # 收集所有问题
        all_issues = []
        quality_scores = []
        
        for result in successful_results:
            all_issues.extend(result.get('issues', []))
            quality_scores.append(result.get('quality_score', 0))
        
        # 统计问题类型
        issue_types = {}
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in all_issues:
            issue_type = issue.get('issue_type', 'unknown')
            severity = issue.get('severity', 'unknown')
            
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # 找出最常见的问题
        most_common_issues = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'total_issues': len(all_issues),
            'issue_type_distribution': issue_types,
            'severity_distribution': severity_counts,
            'most_common_issues': most_common_issues,
            'images_with_issues': len([r for r in successful_results if r.get('issues_detected', 0) > 0]),
            'average_issues_per_image': len(all_issues) / len(successful_results) if successful_results else 0
        }
    
    def _print_diagnostic_summary(self, stats: Dict[str, Any]):
        """
        打印诊断摘要
        """
        print("\n" + "="*60)
        print("OCR问题诊断报告摘要")
        print("="*60)
        
        if 'error' in stats:
            print(f"错误: {stats['error']}")
            return
        
        print(f"平均质量评分: {stats['average_quality_score']:.2f}/100")
        print(f"总问题数: {stats['total_issues']}")
        print(f"有问题的图像数: {stats['images_with_issues']}")
        print(f"平均每图像问题数: {stats['average_issues_per_image']:.2f}")
        
        print("\n问题严重程度分布:")
        for severity, count in stats['severity_distribution'].items():
            print(f"  {severity}: {count}")
        
        print("\n最常见问题:")
        for issue_type, count in stats['most_common_issues']:
            print(f"  {issue_type}: {count}")
        
        print("="*60)

def main():
    """
    主函数
    """
    diagnostics = OCRIssueDiagnostics()
    
    # 测试单张图像（如果存在）
    test_image_path = "test_images/nutrition_label.jpg"
    if os.path.exists(test_image_path):
        print(f"诊断单张图像: {test_image_path}")
        result = diagnostics.diagnose_image(test_image_path)
        print(f"诊断结果: 发现 {result.get('issues_detected', 0)} 个问题")
        
        # 尝试自动修复
        print("\n尝试自动修复...")
        fix_result = diagnostics.apply_automatic_fixes(test_image_path)
        if 'improvement' in fix_result:
            improvement = fix_result['improvement']
            print(f"修复效果: 置信度改进 {improvement['confidence_improvement']:.3f}")
    
    # 测试图像目录（如果存在）
    test_dir = "test_images"
    if os.path.exists(test_dir):
        print(f"\n批量诊断图像目录: {test_dir}")
        image_files = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
        for image_file in image_files:
            diagnostics.diagnose_image(str(image_file))
    
    # 生成报告
    diagnostics.generate_diagnostic_report()
    
    print("\nOCR问题诊断完成！")

if __name__ == "__main__":
    main()