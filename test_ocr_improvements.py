#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR改进效果测试脚本
综合测试所有OCR改进功能的效果
"""

import cv2
import numpy as np
import os
import sys
import logging
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.simple_information_extractor import SimpleInformationExtractor
from core.enhanced_image_processor import EnhancedImageProcessor
from core.chinese_ocr_optimizer import ChineseOCROptimizer
from ocr_issue_diagnostics import OCRIssueDiagnostics
from ocr_repair_tool import OCRRepairTool
from coordinate_accuracy_validator import CoordinateAccuracyValidator

class OCRImprovementTester:
    """
    OCR改进效果测试器
    """
    
    def __init__(self):
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self.extractor = SimpleInformationExtractor()
        self.image_processor = EnhancedImageProcessor()
        self.chinese_optimizer = ChineseOCROptimizer()
        self.diagnostics = OCRIssueDiagnostics()
        self.repair_tool = OCRRepairTool()
        self.validator = CoordinateAccuracyValidator()
        
        # 测试结果
        self.test_results = []
        
        # 设置中文字体（用于可视化）
        self.chinese_font = self._setup_chinese_font()
    
    def _setup_chinese_font(self):
        """
        设置中文字体
        """
        try:
            # 尝试常见的中文字体
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',  # 黑体
                'C:/Windows/Fonts/simsun.ttc',  # 宋体
                'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return FontProperties(fname=font_path)
            
            # 如果没有找到中文字体，使用默认字体
            return FontProperties()
            
        except:
            return FontProperties()
    
    def comprehensive_test(self, image_path: str) -> Dict[str, Any]:
        """
        综合测试单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            综合测试结果
        """
        try:
            self.logger.info(f"开始综合测试: {image_path}")
            start_time = time.time()
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 1. 基础OCR测试
            baseline_result = self._test_baseline_ocr(image)
            
            # 2. 增强OCR测试
            enhanced_result = self._test_enhanced_ocr(image)
            
            # 3. 问题诊断
            diagnostic_result = self.diagnostics.diagnose_image(image_path)
            
            # 4. 自动修复测试
            repair_result = self.repair_tool.repair_image_and_extract(image_path)
            
            # 5. 坐标精度验证
            coordinate_result = self.validator.validate_image_coordinates(image_path)
            
            # 6. 性能对比
            performance_comparison = self._compare_performance(
                baseline_result, enhanced_result, repair_result
            )
            
            # 7. 质量评估
            quality_assessment = self._assess_quality(
                baseline_result, enhanced_result, repair_result, diagnostic_result
            )
            
            # 构建综合结果
            comprehensive_result = {
                'image_path': image_path,
                'image_info': {
                    'size': image.shape,
                    'channels': len(image.shape)
                },
                'baseline_ocr': baseline_result,
                'enhanced_ocr': enhanced_result,
                'diagnostic': diagnostic_result,
                'repair': repair_result,
                'coordinate_validation': coordinate_result,
                'performance_comparison': performance_comparison,
                'quality_assessment': quality_assessment,
                'total_processing_time': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.test_results.append(comprehensive_result)
            
            self.logger.info(f"综合测试完成，总耗时: {comprehensive_result['total_processing_time']:.2f}秒")
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"综合测试失败: {e}")
            return {'error': str(e), 'image_path': image_path}
    
    def _test_baseline_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        测试基础OCR
        """
        try:
            start_time = time.time()
            
            # 使用基础提取器
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            result = self.extractor.extract_comprehensive_info(image, detection_results)
            
            # 计算统计信息
            text_regions = result.get('text_regions', [])
            total_text_length = sum(len(r.get('text', '')) for r in text_regions)
            avg_confidence = np.mean([r.get('confidence', 0) for r in text_regions]) if text_regions else 0
            
            return {
                'result': result,
                'statistics': {
                    'regions_detected': len(text_regions),
                    'total_text_length': total_text_length,
                    'average_confidence': avg_confidence,
                    'processing_time': time.time() - start_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"基础OCR测试失败: {e}")
            return {'error': str(e)}
    
    def _test_enhanced_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        测试增强OCR
        """
        try:
            start_time = time.time()
            
            # 使用增强的图像预处理
            enhanced_image = self.image_processor.enhance_for_ocr(image, 'nutrition')
            
            # 使用中文OCR优化器
            enhanced_results = []
            
            # 检测文本区域
            # 先进行基础检测获取detection_results
            detection_results = {'regions': [], 'barcodes': []}
            detection_result = self.extractor.extract_comprehensive_info(image, detection_results)
            text_regions = detection_result.get('text_regions', [])
            
            for region in text_regions:
                bbox = region.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                    
                    # 提取区域
                    region_image = image[y1:y2, x1:x2]
                    if region_image.size > 0:
                        # 使用中文OCR优化器
                        ocr_result = self.chinese_optimizer.recognize_text_multi_scale(region_image)
                        
                        if ocr_result:
                            enhanced_region = region.copy()
                            enhanced_region.update(ocr_result)
                            enhanced_results.append(enhanced_region)
                        else:
                            enhanced_results.append(region)
                    else:
                        enhanced_results.append(region)
                else:
                    enhanced_results.append(region)
            
            # 构建增强结果
            enhanced_result = detection_result.copy()
            enhanced_result['text_regions'] = enhanced_results
            
            # 计算统计信息
            total_text_length = sum(len(r.get('text', '')) for r in enhanced_results)
            avg_confidence = np.mean([r.get('confidence', 0) for r in enhanced_results]) if enhanced_results else 0
            chinese_char_count = sum(len([c for c in r.get('text', '') if '\u4e00' <= c <= '\u9fff']) for r in enhanced_results)
            
            return {
                'result': enhanced_result,
                'statistics': {
                    'regions_detected': len(enhanced_results),
                    'total_text_length': total_text_length,
                    'average_confidence': avg_confidence,
                    'chinese_char_count': chinese_char_count,
                    'processing_time': time.time() - start_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"增强OCR测试失败: {e}")
            return {'error': str(e)}
    
    def _compare_performance(self, baseline: Dict, enhanced: Dict, repair: Dict) -> Dict[str, Any]:
        """
        比较性能
        """
        try:
            baseline_stats = baseline.get('statistics', {})
            enhanced_stats = enhanced.get('statistics', {})
            repair_comparison = repair.get('comparison', {})
            
            # 文本长度比较
            baseline_text_length = baseline_stats.get('total_text_length', 0)
            enhanced_text_length = enhanced_stats.get('total_text_length', 0)
            
            # 置信度比较
            baseline_confidence = baseline_stats.get('average_confidence', 0)
            enhanced_confidence = enhanced_stats.get('average_confidence', 0)
            
            # 处理时间比较
            baseline_time = baseline_stats.get('processing_time', 0)
            enhanced_time = enhanced_stats.get('processing_time', 0)
            
            return {
                'text_length_improvement': {
                    'baseline_to_enhanced': enhanced_text_length - baseline_text_length,
                    'enhanced_percentage': ((enhanced_text_length - baseline_text_length) / max(baseline_text_length, 1)) * 100,
                    'repair_improvement': repair_comparison.get('text_length_change', 0)
                },
                'confidence_improvement': {
                    'baseline_to_enhanced': enhanced_confidence - baseline_confidence,
                    'enhanced_percentage': ((enhanced_confidence - baseline_confidence) / max(baseline_confidence, 0.1)) * 100,
                    'repair_improvement': repair_comparison.get('confidence_improvement', 0)
                },
                'processing_time': {
                    'baseline': baseline_time,
                    'enhanced': enhanced_time,
                    'time_overhead': enhanced_time - baseline_time,
                    'overhead_percentage': ((enhanced_time - baseline_time) / max(baseline_time, 0.1)) * 100
                },
                'chinese_recognition': {
                    'chinese_chars_detected': enhanced_stats.get('chinese_char_count', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"性能比较失败: {e}")
            return {'error': str(e)}
    
    def _assess_quality(self, baseline: Dict, enhanced: Dict, repair: Dict, diagnostic: Dict) -> Dict[str, Any]:
        """
        评估质量
        """
        try:
            # 基础质量评分
            baseline_score = self._calculate_quality_score(baseline.get('result', {}))
            enhanced_score = self._calculate_quality_score(enhanced.get('result', {}))
            
            # 诊断质量评分
            diagnostic_score = diagnostic.get('quality_score', 0)
            
            # 修复效果评分
            repair_stats = repair.get('comparison', {})
            repair_score = baseline_score + (repair_stats.get('confidence_improvement', 0) * 100)
            
            # 问题检测
            issues_detected = diagnostic.get('issues_detected', 0)
            
            return {
                'quality_scores': {
                    'baseline': baseline_score,
                    'enhanced': enhanced_score,
                    'diagnostic': diagnostic_score,
                    'repair': repair_score
                },
                'improvements': {
                    'baseline_to_enhanced': enhanced_score - baseline_score,
                    'baseline_to_repair': repair_score - baseline_score,
                    'enhanced_to_repair': repair_score - enhanced_score
                },
                'issues_analysis': {
                    'total_issues': issues_detected,
                    'issues_per_region': issues_detected / max(len(baseline.get('result', {}).get('text_regions', [])), 1)
                },
                'overall_assessment': self._get_overall_assessment(baseline_score, enhanced_score, repair_score, issues_detected)
            }
            
        except Exception as e:
            self.logger.error(f"质量评估失败: {e}")
            return {'error': str(e)}
    
    def _calculate_quality_score(self, ocr_result: Dict) -> float:
        """
        计算OCR结果的质量评分
        """
        try:
            text_regions = ocr_result.get('text_regions', [])
            
            if not text_regions:
                return 0.0
            
            # 基础评分
            base_score = 50.0
            
            # 置信度评分 (0-30分)
            confidences = [r.get('confidence', 0) for r in text_regions]
            avg_confidence = np.mean(confidences) if confidences else 0
            confidence_score = avg_confidence * 30
            
            # 文本长度评分 (0-20分)
            total_text_length = sum(len(r.get('text', '')) for r in text_regions)
            text_score = min(20.0, total_text_length / 10)  # 每10个字符1分，最多20分
            
            # 区域数量评分 (0-10分)
            region_score = min(10.0, len(text_regions) * 2)  # 每个区域2分，最多10分
            
            total_score = base_score + confidence_score + text_score + region_score
            
            return min(100.0, total_score)
            
        except:
            return 0.0
    
    def _get_overall_assessment(self, baseline_score: float, enhanced_score: float, repair_score: float, issues_count: int) -> str:
        """
        获取整体评估
        """
        improvement = max(enhanced_score - baseline_score, repair_score - baseline_score)
        
        if improvement > 20:
            return "显著改进"
        elif improvement > 10:
            return "明显改进"
        elif improvement > 5:
            return "轻微改进"
        elif improvement > 0:
            return "微小改进"
        else:
            return "无明显改进"
    
    def visualize_results(self, test_result: Dict, output_path: str = None):
        """
        可视化测试结果
        
        Args:
            test_result: 测试结果
            output_path: 输出路径
        """
        try:
            image_path = test_result.get('image_path')
            if not image_path or not os.path.exists(image_path):
                self.logger.error("无法找到图像文件")
                return
            
            # 加载图像
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'OCR改进效果对比 - {Path(image_path).name}', fontproperties=self.chinese_font, fontsize=16)
            
            # 1. 原始图像 + 基础OCR结果
            ax1 = axes[0, 0]
            ax1.imshow(image_rgb)
            ax1.set_title('基础OCR结果', fontproperties=self.chinese_font)
            self._draw_ocr_boxes(ax1, test_result.get('baseline_ocr', {}).get('result', {}), 'red')
            
            # 2. 原始图像 + 增强OCR结果
            ax2 = axes[0, 1]
            ax2.imshow(image_rgb)
            ax2.set_title('增强OCR结果', fontproperties=self.chinese_font)
            self._draw_ocr_boxes(ax2, test_result.get('enhanced_ocr', {}).get('result', {}), 'green')
            
            # 3. 原始图像 + 修复OCR结果
            ax3 = axes[1, 0]
            ax3.imshow(image_rgb)
            ax3.set_title('修复OCR结果', fontproperties=self.chinese_font)
            repair_result = test_result.get('repair', {}).get('repaired_result', {})
            self._draw_ocr_boxes(ax3, repair_result, 'blue')
            
            # 4. 性能对比图表
            ax4 = axes[1, 1]
            self._draw_performance_chart(ax4, test_result.get('performance_comparison', {}), test_result.get('quality_assessment', {}))
            
            # 移除坐标轴
            for ax in axes.flat[:3]:
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            # 保存图像
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"可视化结果已保存到: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"可视化失败: {e}")
    
    def _draw_ocr_boxes(self, ax, ocr_result: Dict, color: str):
        """
        绘制OCR边界框
        """
        try:
            text_regions = ocr_result.get('text_regions', [])
            
            for i, region in enumerate(text_regions):
                bbox = region.get('bbox', [])
                text = region.get('text', '').strip()
                confidence = region.get('confidence', 0)
                
                if len(bbox) >= 4 and text:
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # 绘制边界框
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加文本标签
                    label = f"{text[:10]}{'...' if len(text) > 10 else ''} ({confidence:.2f})"
                    ax.text(x1, y1-5, label, fontproperties=self.chinese_font, 
                           fontsize=8, color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
        except Exception as e:
            self.logger.debug(f"绘制OCR框失败: {e}")
    
    def _draw_performance_chart(self, ax, performance: Dict, quality: Dict):
        """
        绘制性能对比图表
        """
        try:
            # 准备数据
            categories = ['基础OCR', '增强OCR', '修复OCR']
            quality_scores = [
                quality.get('quality_scores', {}).get('baseline', 0),
                quality.get('quality_scores', {}).get('enhanced', 0),
                quality.get('quality_scores', {}).get('repair', 0)
            ]
            
            # 绘制柱状图
            bars = ax.bar(categories, quality_scores, color=['red', 'green', 'blue'], alpha=0.7)
            
            # 添加数值标签
            for bar, score in zip(bars, quality_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}', ha='center', va='bottom', fontproperties=self.chinese_font)
            
            ax.set_title('质量评分对比', fontproperties=self.chinese_font)
            ax.set_ylabel('质量评分', fontproperties=self.chinese_font)
            ax.set_ylim(0, 100)
            
            # 设置中文标签
            ax.set_xticklabels(categories, fontproperties=self.chinese_font)
            
        except Exception as e:
            self.logger.debug(f"绘制性能图表失败: {e}")
    
    def batch_test(self, input_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """
        批量测试
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            批量测试结果
        """
        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise ValueError(f"输入目录不存在: {input_dir}")
            
            # 查找图像文件
            image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
            
            if not image_files:
                raise ValueError(f"在目录 {input_dir} 中未找到图像文件")
            
            self.logger.info(f"开始批量测试 {len(image_files)} 个图像文件")
            
            # 批量处理
            batch_results = []
            for i, image_file in enumerate(image_files):
                self.logger.info(f"处理第 {i+1}/{len(image_files)} 个文件: {image_file.name}")
                result = self.comprehensive_test(str(image_file))
                batch_results.append(result)
                
                # 生成可视化结果
                if output_dir and 'error' not in result:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    viz_path = output_path / f"{image_file.stem}_comparison.png"
                    self.visualize_results(result, str(viz_path))
            
            # 计算批量统计
            batch_stats = self._calculate_batch_test_statistics(batch_results)
            
            # 保存结果
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # 保存详细结果
                results_file = output_path / 'comprehensive_test_results.json'
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'batch_statistics': batch_stats,
                        'detailed_results': batch_results
                    }, f, ensure_ascii=False, indent=2)
                
                # 生成汇总报告
                self._generate_summary_report(batch_stats, output_path / 'test_summary_report.txt')
                
                self.logger.info(f"批量测试结果已保存到: {output_path}")
            
            return {
                'batch_statistics': batch_stats,
                'processed_files': len(image_files),
                'results': batch_results
            }
            
        except Exception as e:
            self.logger.error(f"批量测试失败: {e}")
            return {'error': str(e)}
    
    def _calculate_batch_test_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算批量测试统计信息
        """
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {'error': '没有成功的测试结果'}
        
        # 收集性能数据
        baseline_scores = []
        enhanced_scores = []
        repair_scores = []
        confidence_improvements = []
        text_length_improvements = []
        processing_times = []
        
        for result in successful_results:
            quality = result.get('quality_assessment', {})
            performance = result.get('performance_comparison', {})
            
            quality_scores = quality.get('quality_scores', {})
            baseline_scores.append(quality_scores.get('baseline', 0))
            enhanced_scores.append(quality_scores.get('enhanced', 0))
            repair_scores.append(quality_scores.get('repair', 0))
            
            conf_improvement = performance.get('confidence_improvement', {})
            confidence_improvements.append(conf_improvement.get('baseline_to_enhanced', 0))
            
            text_improvement = performance.get('text_length_improvement', {})
            text_length_improvements.append(text_improvement.get('baseline_to_enhanced', 0))
            
            processing_times.append(result.get('total_processing_time', 0))
        
        return {
            'total_images': len(successful_results),
            'average_scores': {
                'baseline': np.mean(baseline_scores),
                'enhanced': np.mean(enhanced_scores),
                'repair': np.mean(repair_scores)
            },
            'score_improvements': {
                'baseline_to_enhanced': np.mean(enhanced_scores) - np.mean(baseline_scores),
                'baseline_to_repair': np.mean(repair_scores) - np.mean(baseline_scores)
            },
            'average_improvements': {
                'confidence': np.mean(confidence_improvements),
                'text_length': np.mean(text_length_improvements)
            },
            'performance': {
                'average_processing_time': np.mean(processing_times),
                'total_processing_time': sum(processing_times)
            },
            'success_rate': len(successful_results) / len(results) if results else 0
        }
    
    def _generate_summary_report(self, stats: Dict, output_path: str):
        """
        生成汇总报告
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("OCR改进效果测试汇总报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"测试图像数: {stats.get('total_images', 0)}\n")
                f.write(f"成功率: {stats.get('success_rate', 0):.2%}\n\n")
                
                f.write("质量评分对比:\n")
                avg_scores = stats.get('average_scores', {})
                f.write(f"  基础OCR平均分: {avg_scores.get('baseline', 0):.2f}\n")
                f.write(f"  增强OCR平均分: {avg_scores.get('enhanced', 0):.2f}\n")
                f.write(f"  修复OCR平均分: {avg_scores.get('repair', 0):.2f}\n\n")
                
                f.write("改进效果:\n")
                improvements = stats.get('score_improvements', {})
                f.write(f"  基础到增强改进: {improvements.get('baseline_to_enhanced', 0):.2f}分\n")
                f.write(f"  基础到修复改进: {improvements.get('baseline_to_repair', 0):.2f}分\n\n")
                
                f.write("平均改进指标:\n")
                avg_improvements = stats.get('average_improvements', {})
                f.write(f"  置信度改进: {avg_improvements.get('confidence', 0):.3f}\n")
                f.write(f"  文本长度改进: {avg_improvements.get('text_length', 0):.1f}字符\n\n")
                
                f.write("性能统计:\n")
                performance = stats.get('performance', {})
                f.write(f"  平均处理时间: {performance.get('average_processing_time', 0):.2f}秒\n")
                f.write(f"  总处理时间: {performance.get('total_processing_time', 0):.2f}秒\n")
                
            self.logger.info(f"汇总报告已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"生成汇总报告失败: {e}")

def main():
    """
    主函数
    """
    tester = OCRImprovementTester()
    
    # 测试单张图像
    test_image_path = "test_images/nutrition_label.jpg"
    if os.path.exists(test_image_path):
        print(f"综合测试单张图像: {test_image_path}")
        result = tester.comprehensive_test(test_image_path)
        
        if 'error' not in result:
            # 显示测试结果摘要
            quality = result.get('quality_assessment', {})
            performance = result.get('performance_comparison', {})
            
            print("\n测试结果摘要:")
            print(f"  质量评分改进: {quality.get('improvements', {}).get('baseline_to_enhanced', 0):.2f}")
            print(f"  置信度改进: {performance.get('confidence_improvement', {}).get('baseline_to_enhanced', 0):.3f}")
            print(f"  整体评估: {quality.get('overall_assessment', '未知')}")
            
            # 生成可视化
            tester.visualize_results(result, "single_test_result.png")
    
    # 批量测试
    test_dir = "test_images"
    if os.path.exists(test_dir):
        print(f"\n批量测试目录: {test_dir}")
        batch_result = tester.batch_test(test_dir, "comprehensive_test_results")
        
        if 'batch_statistics' in batch_result:
            stats = batch_result['batch_statistics']
            print(f"\n批量测试统计:")
            print(f"  测试图像数: {stats.get('total_images', 0)}")
            print(f"  平均质量改进: {stats.get('score_improvements', {}).get('baseline_to_enhanced', 0):.2f}")
            print(f"  平均置信度改进: {stats.get('average_improvements', {}).get('confidence', 0):.3f}")
    
    print("\nOCR改进效果测试完成！")

if __name__ == "__main__":
    main()