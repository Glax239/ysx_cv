# -*- coding: utf-8 -*-
"""
文件管理工具
File management utilities
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import shutil

from config import OUTPUT_CONFIG

class FileManager:
    """文件管理器"""
    
    def __init__(self):
        """初始化文件管理器"""
        self.logger = logging.getLogger(__name__)
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            OUTPUT_CONFIG['results_dir'],
            OUTPUT_CONFIG['temp_dir'],
            OUTPUT_CONFIG['log_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_detection_results(self, results: Dict, image_path: str, 
                             output_name: Optional[str] = None) -> str:
        """
        保存检测结果
        
        Args:
            results: 检测结果字典
            image_path: 原始图像路径
            output_name: 输出文件名（可选）
            
        Returns:
            保存的文件路径
        """
        try:
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = Path(image_path).stem
                output_name = f"{image_name}_detection_{timestamp}.json"
            
            output_path = Path(OUTPUT_CONFIG['results_dir']) / output_name
            
            # 准备保存的数据
            save_data = {
                'metadata': {
                    'original_image': str(image_path),
                    'detection_time': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'results': self._make_json_serializable(results)
            }
            
            # 保存JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"检测结果已保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"保存检测结果失败: {e}")
            raise
    
    def save_extraction_results(self, extraction_data: Dict, image_path: str,
                              output_name: Optional[str] = None) -> str:
        """
        保存信息提取结果
        
        Args:
            extraction_data: 提取的信息数据
            image_path: 原始图像路径
            output_name: 输出文件名（可选）
            
        Returns:
            保存的文件路径
        """
        try:
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = Path(image_path).stem
                output_name = f"{image_name}_extraction_{timestamp}.json"
            
            output_path = Path(OUTPUT_CONFIG['results_dir']) / output_name
            
            # 准备保存的数据
            save_data = {
                'metadata': {
                    'original_image': str(image_path),
                    'extraction_time': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'extraction_results': self._make_json_serializable(extraction_data)
            }
            
            # 保存JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"提取结果已保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"保存提取结果失败: {e}")
            raise
    
    def save_analysis_report(self, analysis_data: Dict, output_name: Optional[str] = None) -> str:
        """
        保存分析报告
        
        Args:
            analysis_data: 分析数据
            output_name: 输出文件名（可选）
            
        Returns:
            保存的文件路径
        """
        try:
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"analysis_report_{timestamp}.json"
            
            output_path = Path(OUTPUT_CONFIG['results_dir']) / output_name
            
            # 准备保存的数据
            save_data = {
                'metadata': {
                    'report_time': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'analysis': self._make_json_serializable(analysis_data)
            }
            
            # 保存JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分析报告已保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"保存分析报告失败: {e}")
            raise
    
    def export_to_csv(self, data: List[Dict], output_path: str, 
                     fieldnames: Optional[List[str]] = None) -> str:
        """
        导出数据到CSV文件
        
        Args:
            data: 要导出的数据列表
            output_path: 输出文件路径
            fieldnames: CSV字段名列表（可选）
            
        Returns:
            保存的文件路径
        """
        try:
            if not data:
                raise ValueError("没有数据可导出")
            
            # 如果没有指定字段名，使用第一条记录的键
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in data:
                    # 确保所有值都是可序列化的
                    clean_row = {}
                    for key, value in row.items():
                        if key in fieldnames:
                            clean_row[key] = str(value) if value is not None else ''
                    writer.writerow(clean_row)
            
            self.logger.info(f"数据已导出到CSV: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"导出CSV失败: {e}")
            raise
    
    def load_results(self, file_path: str) -> Dict:
        """
        加载保存的结果文件
        
        Args:
            file_path: 结果文件路径
            
        Returns:
            加载的结果数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"结果文件已加载: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"加载结果文件失败: {e}")
            raise
    
    def clean_temp_files(self, max_age_hours: int = 24):
        """
        清理临时文件
        
        Args:
            max_age_hours: 文件最大保留时间（小时）
        """
        try:
            temp_dir = Path(OUTPUT_CONFIG['temp_dir'])
            if not temp_dir.exists():
                return
            
            current_time = datetime.now()
            cleaned_count = 0
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    # 获取文件修改时间
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age_hours = (current_time - file_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        file_path.unlink()
                        cleaned_count += 1
            
            self.logger.info(f"清理了 {cleaned_count} 个临时文件")
            
        except Exception as e:
            self.logger.error(f"清理临时文件失败: {e}")
    
    def backup_results(self, backup_dir: Optional[str] = None):
        """
        备份结果文件
        
        Args:
            backup_dir: 备份目录（可选）
        """
        try:
            if backup_dir is None:
                backup_dir = Path(OUTPUT_CONFIG['results_dir']).parent / 'backup'
            
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 创建带时间戳的备份目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_path / f"backup_{timestamp}"
            
            # 复制结果目录
            shutil.copytree(OUTPUT_CONFIG['results_dir'], backup_subdir)
            
            self.logger.info(f"结果已备份到: {backup_subdir}")
            return str(backup_subdir)
            
        except Exception as e:
            self.logger.error(f"备份失败: {e}")
            raise
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def get_file_info(self, file_path: str) -> Dict:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            stat = path.stat()
            
            return {
                'name': path.name,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'is_file': path.is_file(),
                'is_dir': path.is_dir(),
                'extension': path.suffix,
                'absolute_path': str(path.absolute())
            }
            
        except Exception as e:
            self.logger.error(f"获取文件信息失败: {e}")
            raise
