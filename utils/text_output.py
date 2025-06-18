# -*- coding: utf-8 -*-
"""
文本输出模块
Text Output Module
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

class TextOutputManager:
    """文本输出管理器"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化文本输出管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 创建子目录
        (self.output_dir / "text_results").mkdir(exist_ok=True)
        (self.output_dir / "json_results").mkdir(exist_ok=True)
        (self.output_dir / "csv_results").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
    
    def save_text_results(self, extraction_results: Dict, image_path: str = None) -> str:
        """
        保存文本识别结果为纯文本文件
        
        Args:
            extraction_results: 提取结果
            image_path: 图像路径
            
        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"text_results_{timestamp}.txt"
            filepath = self.output_dir / "text_results" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("智能商品识别系统 - 文本识别结果\n")
                f.write("=" * 60 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if image_path:
                    f.write(f"图像文件: {image_path}\n")
                f.write("\n")
                
                # OCR文本识别结果
                text_info = extraction_results.get('text_info', [])
                if text_info:
                    f.write("📝 OCR文本识别结果:\n")
                    f.write("-" * 40 + "\n")
                    for i, text in enumerate(text_info, 1):
                        f.write(f"{i}. 识别文本: {text['text']}\n")
                        f.write(f"   置信度: {text['confidence']:.3f}\n")
                        f.write(f"   OCR引擎: {text.get('engine', 'unknown')}\n")
                        if 'bbox' in text:
                            bbox = text['bbox']
                            f.write(f"   位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n")
                        f.write("\n")
                else:
                    f.write("📝 OCR文本识别结果: 未检测到文本\n\n")
                
                # 条形码信息
                barcodes = extraction_results.get('barcodes', [])
                if barcodes:
                    f.write("🔢 条形码信息:\n")
                    f.write("-" * 40 + "\n")
                    for i, barcode in enumerate(barcodes, 1):
                        f.write(f"{i}. 条形码: {barcode['data']}\n")
                        f.write(f"   类型: {barcode['type']}\n")
                        if 'bbox' in barcode:
                            bbox = barcode['bbox']
                            f.write(f"   位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n")
                        f.write("\n")
                
                # 营养信息
                nutrition_info = extraction_results.get('nutrition_info', {})
                if nutrition_info and any(v is not None for v in nutrition_info.values() if v != 'raw_texts'):
                    f.write("🍎 营养成分信息:\n")
                    f.write("-" * 40 + "\n")
                    if nutrition_info.get('energy'):
                        f.write(f"能量: {nutrition_info['energy']} kJ\n")
                    if nutrition_info.get('protein'):
                        f.write(f"蛋白质: {nutrition_info['protein']} g\n")
                    if nutrition_info.get('fat'):
                        f.write(f"脂肪: {nutrition_info['fat']} g\n")
                    if nutrition_info.get('carbohydrate'):
                        f.write(f"碳水化合物: {nutrition_info['carbohydrate']} g\n")
                    if nutrition_info.get('sodium'):
                        f.write(f"钠: {nutrition_info['sodium']} mg\n")
                    if nutrition_info.get('sugar'):
                        f.write(f"糖: {nutrition_info['sugar']} g\n")
                    f.write("\n")
                

                
                # 所有识别的文本内容（纯文本）
                f.write("📄 所有识别的文本内容:\n")
                f.write("-" * 40 + "\n")
                all_texts = []
                
                # 收集所有文本
                for text in text_info:
                    all_texts.append(text['text'])
                
                if nutrition_info.get('raw_texts'):
                    all_texts.extend(nutrition_info['raw_texts'])
                
                # 去重并输出
                unique_texts = list(dict.fromkeys(all_texts))  # 保持顺序的去重
                if unique_texts:
                    for text in unique_texts:
                        if text.strip():
                            f.write(f"• {text.strip()}\n")
                else:
                    f.write("未识别到任何文本内容\n")
                
                f.write("\n")
                f.write("=" * 60 + "\n")
                f.write("报告结束\n")
            
            self.logger.info(f"文本结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存文本结果失败: {e}")
            return ""
    
    def save_json_results(self, extraction_results: Dict, detection_results: Dict = None, 
                         image_path: str = None) -> str:
        """
        保存完整结果为JSON文件
        
        Args:
            extraction_results: 提取结果
            detection_results: 检测结果
            image_path: 图像路径
            
        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complete_results_{timestamp}.json"
            filepath = self.output_dir / "json_results" / filename
            
            complete_results = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'extraction_results': extraction_results,
                'detection_results': detection_results or {}
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存JSON结果失败: {e}")
            return ""
    
    def save_csv_results(self, extraction_results: Dict, image_path: str = None) -> str:
        """
        保存文本识别结果为CSV文件
        
        Args:
            extraction_results: 提取结果
            image_path: 图像路径
            
        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"text_results_{timestamp}.csv"
            filepath = self.output_dir / "csv_results" / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # 写入标题行
                writer.writerow(['序号', '识别文本', '置信度', 'OCR引擎', '位置X1', '位置Y1', '位置X2', '位置Y2'])
                
                # 写入文本识别结果
                text_info = extraction_results.get('text_info', [])
                for i, text in enumerate(text_info, 1):
                    bbox = text.get('bbox', (0, 0, 0, 0))
                    writer.writerow([
                        i,
                        text['text'],
                        f"{text['confidence']:.3f}",
                        text.get('engine', 'unknown'),
                        bbox[0], bbox[1], bbox[2], bbox[3]
                    ])
            
            self.logger.info(f"CSV结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存CSV结果失败: {e}")
            return ""
    
    def generate_summary_report(self, extraction_results: Dict, detection_results: Dict = None,
                               image_path: str = None) -> str:
        """
        生成汇总报告
        
        Args:
            extraction_results: 提取结果
            detection_results: 检测结果
            image_path: 图像路径
            
        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.txt"
            filepath = self.output_dir / "reports" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("智能商品识别系统 - 汇总报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if image_path:
                    f.write(f"图像文件: {image_path}\n")
                f.write("\n")
                
                # 检测统计
                if detection_results:
                    f.write("🔍 检测统计:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"检测到商品数量: {len(detection_results.get('products', []))}\n")
                f.write(f"检测到区域数量: {len(detection_results.get('regions', []))}\n")
                f.write(f"检测到文本区域数量: {len(detection_results.get('texts', []))}\n")
                f.write("\n")
                
                # 信息提取统计
                f.write("📊 信息提取统计:\n")
                f.write("-" * 50 + "\n")
                text_count = len(extraction_results.get('text_info', []))
                barcode_count = len(extraction_results.get('barcodes', []))
                has_nutrition = bool(extraction_results.get('nutrition_info', {}))
                
                f.write(f"识别文本数量: {text_count}\n")
                f.write(f"条形码数量: {barcode_count}\n")
                f.write(f"营养信息: {'有' if has_nutrition else '无'}\n")
                f.write("\n")
                
                # 文本内容摘要
                f.write("📝 文本内容摘要:\n")
                f.write("-" * 50 + "\n")
                all_texts = []
                for text in extraction_results.get('text_info', []):
                    all_texts.append(text['text'])
                
                if all_texts:
                    # 统计字符类型
                    total_chars = sum(len(text) for text in all_texts)
                    chinese_chars = sum(len([c for c in text if '\u4e00' <= c <= '\u9fff']) for text in all_texts)
                    english_chars = sum(len([c for c in text if c.isalpha() and ord(c) < 128]) for text in all_texts)
                    digit_chars = sum(len([c for c in text if c.isdigit()]) for text in all_texts)
                    
                    f.write(f"总字符数: {total_chars}\n")
                    f.write(f"中文字符: {chinese_chars}\n")
                    f.write(f"英文字符: {english_chars}\n")
                    f.write(f"数字字符: {digit_chars}\n")
                    f.write(f"其他字符: {total_chars - chinese_chars - english_chars - digit_chars}\n")
                else:
                    f.write("未识别到文本内容\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write("报告结束\n")
            
            self.logger.info(f"汇总报告已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"生成汇总报告失败: {e}")
            return ""
    
    def get_all_text_content(self, extraction_results: Dict) -> List[str]:
        """
        获取所有识别的文本内容
        
        Args:
            extraction_results: 提取结果
            
        Returns:
            所有文本内容列表
        """
        all_texts = []
        
        # OCR识别的文本
        for text in extraction_results.get('text_info', []):
            all_texts.append(text['text'])
        
        # 营养信息中的原始文本
        nutrition_info = extraction_results.get('nutrition_info', {})
        if nutrition_info.get('raw_texts'):
            all_texts.extend(nutrition_info['raw_texts'])
        

        
        # 去重并过滤空文本
        unique_texts = []
        seen = set()
        for text in all_texts:
            text = text.strip()
            if text and text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        return unique_texts
