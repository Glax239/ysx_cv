#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR改进工具使用示例
展示如何使用所有OCR改进功能
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.simple_information_extractor import SimpleInformationExtractor
from core.enhanced_image_processor import EnhancedImageProcessor
from core.chinese_ocr_optimizer import ChineseOCROptimizer
from ocr_issue_diagnostics import OCRIssueDiagnostics
from ocr_repair_tool import OCRRepairTool
from coordinate_accuracy_validator import CoordinateAccuracyValidator
from test_ocr_improvements import OCRImprovementTester

def example_basic_usage():
    """
    基础使用示例
    """
    print("=== 基础OCR使用示例 ===")
    
    # 创建提取器
    extractor = SimpleInformationExtractor()
    
    # 示例图像路径（请替换为实际图像路径）
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        print("请将营养成分表图像放在 test_images/ 目录下")
        return
    
    # 加载图像
    image = cv2.imread(image_path)
    
    # 提取信息
    # 先进行基础检测获取detection_results
    detection_results = {'regions': [], 'barcodes': []}
    result = extractor.extract_comprehensive_info(image, detection_results)
    
    # 显示结果
    print(f"检测到 {len(result.get('text_regions', []))} 个文本区域")
    
    for i, region in enumerate(result.get('text_regions', [])):
        text = region.get('text', '').strip()
        confidence = region.get('confidence', 0)
        bbox = region.get('bbox', [])
        
        if text:
            print(f"区域 {i+1}: '{text}' (置信度: {confidence:.3f})")
            if len(bbox) >= 4:
                print(f"  坐标: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
    
    print()

def example_enhanced_processing():
    """
    增强处理示例
    """
    print("=== 增强图像处理示例 ===")
    
    # 创建增强处理器
    processor = EnhancedImageProcessor()
    
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        return
    
    # 加载图像
    image = cv2.imread(image_path)
    
    # 不同类型的增强处理
    enhanced_nutrition = processor.enhance_for_ocr(image, 'nutrition')
    enhanced_text = processor.enhance_for_ocr(image, 'text')
    enhanced_general = processor.enhance_for_ocr(image, 'general')
    
    print("已生成不同类型的增强图像:")
    print("- 营养成分表专用增强")
    print("- 普通文本增强")
    print("- 通用增强")
    
    # 保存增强图像（可选）
    output_dir = Path("enhanced_images")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "enhanced_nutrition.jpg"), enhanced_nutrition)
    cv2.imwrite(str(output_dir / "enhanced_text.jpg"), enhanced_text)
    cv2.imwrite(str(output_dir / "enhanced_general.jpg"), enhanced_general)
    
    print(f"增强图像已保存到: {output_dir}")
    print()

def example_chinese_ocr_optimization():
    """
    中文OCR优化示例
    """
    print("=== 中文OCR优化示例 ===")
    
    # 创建中文OCR优化器
    optimizer = ChineseOCROptimizer()
    
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        return
    
    # 加载图像
    image = cv2.imread(image_path)
    
    # 假设我们有一个包含中文的区域
    # 这里使用整个图像作为示例
    result = optimizer.recognize_text_multi_scale(image)
    
    if result:
        print(f"识别文本: '{result.get('text', '')}'") 
        print(f"置信度: {result.get('confidence', 0):.3f}")
        print(f"中文字符数: {result.get('chinese_char_count', 0)}")
        print(f"使用的缩放比例: {result.get('scale_used', 1.0)}")
        print(f"OCR引擎: {result.get('engine', 'unknown')}")
    else:
        print("中文OCR识别失败")
    
    print()

def example_issue_diagnostics():
    """
    问题诊断示例
    """
    print("=== OCR问题诊断示例 ===")
    
    # 创建诊断器
    diagnostics = OCRIssueDiagnostics()
    
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        return
    
    # 诊断图像
    diagnostic_result = diagnostics.diagnose_image(image_path)
    
    if 'error' not in diagnostic_result:
        print(f"质量评分: {diagnostic_result.get('quality_score', 0):.2f}/100")
        print(f"发现问题数: {diagnostic_result.get('issues_detected', 0)}")
        
        # 显示主要问题
        issues = diagnostic_result.get('issues', [])
        if issues:
            print("\n主要问题:")
            for issue in issues[:3]:  # 显示前3个问题
                print(f"- {issue.get('description', '')} (严重程度: {issue.get('severity', '')})")
                print(f"  建议修复: {issue.get('suggested_fix', '')}")
        else:
            print("未发现明显问题")
    else:
        print(f"诊断失败: {diagnostic_result.get('error', '')}")
    
    print()

def example_automatic_repair():
    """
    自动修复示例
    """
    print("=== 自动修复示例 ===")
    
    # 创建修复工具
    repair_tool = OCRRepairTool()
    
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        return
    
    # 执行修复
    repair_result = repair_tool.repair_image_and_extract(image_path)
    
    if 'error' not in repair_result:
        comparison = repair_result.get('comparison', {})
        
        print(f"修复效果:")
        print(f"- 置信度改进: {comparison.get('confidence_improvement', 0):.3f}")
        print(f"- 文本长度变化: {comparison.get('text_length_change', 0)} 字符")
        print(f"- 修复区域数: {comparison.get('regions_repaired', 0)}/{comparison.get('total_regions', 0)}")
        print(f"- 修复率: {comparison.get('repair_rate', 0):.2%}")
        print(f"- 改进百分比: {comparison.get('improvement_percentage', 0):.1f}%")
    else:
        print(f"修复失败: {repair_result.get('error', '')}")
    
    print()

def example_coordinate_validation():
    """
    坐标精度验证示例
    """
    print("=== 坐标精度验证示例 ===")
    
    # 创建验证器
    validator = CoordinateAccuracyValidator()
    
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        return
    
    # 验证坐标精度
    validation_result = validator.validate_single_image(image_path)
    
    if 'error' not in validation_result:
        metrics = validation_result.get('metrics', {})
        
        print(f"坐标精度验证结果:")
        print(f"- 坐标准确率: {metrics.get('coordinate_accuracy', 0):.2%}")
        print(f"- 边界框覆盖率: {metrics.get('bbox_coverage', 0):.2%}")
        print(f"- 解析成功率: {metrics.get('parsing_success_rate', 0):.2%}")
        print(f"- 平均置信度: {metrics.get('average_confidence', 0):.3f}")
        print(f"- 有效区域比例: {metrics.get('valid_region_ratio', 0):.2%}")
    else:
        print(f"验证失败: {validation_result.get('error', '')}")
    
    print()

def example_comprehensive_test():
    """
    综合测试示例
    """
    print("=== 综合测试示例 ===")
    
    # 创建综合测试器
    tester = OCRImprovementTester()
    
    image_path = "test_images/nutrition_label.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在: {image_path}")
        return
    
    # 执行综合测试
    test_result = tester.comprehensive_test(image_path)
    
    if 'error' not in test_result:
        quality = test_result.get('quality_assessment', {})
        performance = test_result.get('performance_comparison', {})
        
        print(f"综合测试结果:")
        
        # 质量评分
        quality_scores = quality.get('quality_scores', {})
        print(f"\n质量评分:")
        print(f"- 基础OCR: {quality_scores.get('baseline', 0):.2f}")
        print(f"- 增强OCR: {quality_scores.get('enhanced', 0):.2f}")
        print(f"- 修复OCR: {quality_scores.get('repair', 0):.2f}")
        
        # 改进效果
        improvements = quality.get('improvements', {})
        print(f"\n改进效果:")
        print(f"- 基础到增强: +{improvements.get('baseline_to_enhanced', 0):.2f}")
        print(f"- 基础到修复: +{improvements.get('baseline_to_repair', 0):.2f}")
        
        # 整体评估
        print(f"\n整体评估: {quality.get('overall_assessment', '未知')}")
        
        # 生成可视化结果
        tester.visualize_results(test_result, "comprehensive_test_visualization.png")
        print(f"\n可视化结果已保存为: comprehensive_test_visualization.png")
    else:
        print(f"综合测试失败: {test_result.get('error', '')}")
    
    print()

def example_batch_processing():
    """
    批量处理示例
    """
    print("=== 批量处理示例 ===")
    
    test_dir = "test_images"
    
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        print("请创建 test_images/ 目录并放入一些图像文件")
        return
    
    # 检查是否有图像文件
    image_files = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    
    if not image_files:
        print(f"在 {test_dir} 目录中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 批量修复
    repair_tool = OCRRepairTool()
    batch_repair_result = repair_tool.batch_repair(test_dir, "batch_repair_results")
    
    if 'statistics' in batch_repair_result:
        stats = batch_repair_result['statistics']
        print(f"\n批量修复结果:")
        print(f"- 处理图像数: {stats.get('total_images_processed', 0)}")
        print(f"- 平均置信度改进: {stats.get('average_confidence_improvement', 0):.3f}")
        print(f"- 总体修复率: {stats.get('overall_repair_rate', 0):.2%}")
        print(f"- 成功率: {stats.get('success_rate', 0):.2%}")
    
    # 批量综合测试
    tester = OCRImprovementTester()
    batch_test_result = tester.batch_test(test_dir, "batch_test_results")
    
    if 'batch_statistics' in batch_test_result:
        stats = batch_test_result['batch_statistics']
        print(f"\n批量测试结果:")
        print(f"- 测试图像数: {stats.get('total_images', 0)}")
        print(f"- 平均质量改进: {stats.get('score_improvements', {}).get('baseline_to_enhanced', 0):.2f}")
        print(f"- 平均处理时间: {stats.get('performance', {}).get('average_processing_time', 0):.2f}秒")
    
    print()

def create_sample_test_structure():
    """
    创建示例测试结构
    """
    print("=== 创建示例测试结构 ===")
    
    # 创建测试目录
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 创建结果目录
    result_dirs = [
        "enhanced_images",
        "batch_repair_results", 
        "batch_test_results",
        "comprehensive_test_results"
    ]
    
    for dir_name in result_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("已创建以下目录结构:")
    print("- test_images/          # 放置测试图像")
    print("- enhanced_images/      # 增强图像输出")
    print("- batch_repair_results/ # 批量修复结果")
    print("- batch_test_results/   # 批量测试结果")
    print("- comprehensive_test_results/ # 综合测试结果")
    
    # 创建README文件
    readme_content = """
# OCR改进工具使用说明

## 目录结构
- test_images/: 放置测试图像（支持 .jpg, .png, .jpeg 格式）
- enhanced_images/: 增强处理后的图像
- batch_repair_results/: 批量修复结果
- batch_test_results/: 批量测试结果
- comprehensive_test_results/: 综合测试结果

## 使用方法
1. 将营养成分表图像放入 test_images/ 目录
2. 运行 python example_usage.py
3. 查看各个目录中的结果文件

## 主要功能
- 基础OCR识别
- 增强图像处理
- 中文OCR优化
- 问题自动诊断
- 智能修复
- 坐标精度验证
- 综合效果测试
"""
    
    with open("README_OCR_Tools.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("\n已创建 README_OCR_Tools.md 使用说明文件")
    print()

def main():
    """
    主函数 - 运行所有示例
    """
    print("OCR改进工具使用示例")
    print("=" * 50)
    print()
    
    # 创建测试结构
    create_sample_test_structure()
    
    # 运行各种示例
    example_basic_usage()
    example_enhanced_processing()
    example_chinese_ocr_optimization()
    example_issue_diagnostics()
    example_automatic_repair()
    example_coordinate_validation()
    example_comprehensive_test()
    example_batch_processing()
    
    print("=" * 50)
    print("所有示例运行完成！")
    print()
    print("提示:")
    print("1. 请将营养成分表图像放入 test_images/ 目录")
    print("2. 重新运行此脚本以查看完整效果")
    print("3. 查看各个结果目录中的输出文件")
    print("4. 阅读 README_OCR_Tools.md 了解详细使用方法")

if __name__ == "__main__":
    main()