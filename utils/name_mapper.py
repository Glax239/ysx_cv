# -*- coding: utf-8 -*-
"""
商品名称映射工具
Product Name Mapping Utility

用于将模型输出的类别ID映射到用户自定义的商品名称
"""

import json
import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

class ProductNameMapper:
    """商品名称映射器"""
    
    def __init__(self, mapping_file: str = None):
        """
        初始化名称映射器

        Args:
            mapping_file: 映射文件路径，默认使用项目根目录下的product_name_mapping.json
        """
        self.logger = logging.getLogger(__name__)

        # 设置默认映射文件路径
        if mapping_file is None:
            project_root = Path(__file__).parent.parent
            # 优先使用config目录中的映射文件
            config_mapping_file = project_root / "config" / "product_name_mapping.json"
            if config_mapping_file.exists():
                mapping_file = config_mapping_file
                self.logger.info(f"使用config目录中的映射文件: {config_mapping_file}")
            else:
                mapping_file = project_root / "product_name_mapping.json"
                self.logger.info(f"使用根目录中的映射文件: {mapping_file}")

        self.mapping_file = mapping_file
        self.mappings = {}
        self.load_mappings()
    
    def load_mappings(self) -> bool:
        """
        加载名称映射配置

        Returns:
            是否成功加载映射配置
        """
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    self.mappings = json.load(f)
                self.logger.info(f"成功加载名称映射配置: {self.mapping_file}")

                # 尝试加载并合并其他位置的映射文件
                self._merge_additional_mappings()
                return True
            else:
                self.logger.warning(f"映射文件不存在: {self.mapping_file}")
                self._create_default_mappings()
                return False
        except Exception as e:
            self.logger.error(f"加载名称映射配置失败: {e}")
            self._create_default_mappings()
            return False

    def _merge_additional_mappings(self):
        """合并其他位置的映射文件"""
        try:
            project_root = Path(__file__).parent.parent

            # 检查根目录和config目录中的映射文件
            additional_files = [
                project_root / "product_name_mapping.json",
                project_root / "config" / "product_name_mapping.json"
            ]

            for additional_file in additional_files:
                if additional_file.exists() and str(additional_file) != str(self.mapping_file):
                    try:
                        with open(additional_file, 'r', encoding='utf-8') as f:
                            additional_mappings = json.load(f)

                        # 合并映射配置
                        for model_name, mappings in additional_mappings.items():
                            if model_name not in self.mappings:
                                self.mappings[model_name] = {}

                            # 合并映射，新的映射会覆盖旧的
                            self.mappings[model_name].update(mappings)

                        self.logger.info(f"成功合并映射文件: {additional_file}")

                    except Exception as e:
                        self.logger.warning(f"合并映射文件失败 {additional_file}: {e}")

        except Exception as e:
            self.logger.error(f"合并额外映射文件时出错: {e}")
    
    def _create_default_mappings(self):
        """创建默认映射配置"""
        self.mappings = {
            "product_detector": {},
        "region_detector": {},
        "text_detector": {}
        }
        self.logger.info("使用默认映射配置")
    
    def save_mappings(self) -> bool:
        """
        保存名称映射配置到文件
        
        Returns:
            是否成功保存
        """
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mappings, f, ensure_ascii=False, indent=2)
            self.logger.info(f"成功保存名称映射配置: {self.mapping_file}")
            return True
        except Exception as e:
            self.logger.error(f"保存名称映射配置失败: {e}")
            return False
    
    def map_class_name(self, model_name: str, class_id: int, original_name: str) -> str:
        """
        映射类别名称
        
        Args:
            model_name: 模型名称 (如 'product_detector')
            class_id: 类别ID
            original_name: 原始类别名称
            
        Returns:
            映射后的名称，如果没有找到映射则返回原始名称
        """
        try:
            # 获取对应模型的映射配置
            model_mappings = self.mappings.get(model_name, {})
            
            # 尝试使用class_id进行映射
            class_id_str = str(class_id)
            if class_id_str in model_mappings:
                mapped_name = model_mappings[class_id_str]
                self.logger.debug(f"映射 {model_name}[{class_id}] '{original_name}' -> '{mapped_name}'")
                return mapped_name
            
            # 尝试使用原始名称进行映射
            if original_name in model_mappings:
                mapped_name = model_mappings[original_name]
                self.logger.debug(f"映射 {model_name}['{original_name}'] -> '{mapped_name}'")
                return mapped_name
            
            # 如果没有找到映射，返回原始名称
            self.logger.debug(f"未找到映射 {model_name}[{class_id}] '{original_name}'，使用原始名称")
            return original_name
            
        except Exception as e:
            self.logger.error(f"映射类别名称时出错: {e}")
            return original_name
    
    def add_mapping(self, model_name: str, class_id: int, custom_name: str) -> bool:
        """
        添加新的名称映射
        
        Args:
            model_name: 模型名称
            class_id: 类别ID
            custom_name: 自定义名称
            
        Returns:
            是否成功添加
        """
        try:
            if model_name not in self.mappings:
                self.mappings[model_name] = {}
            
            self.mappings[model_name][str(class_id)] = custom_name
            self.logger.info(f"添加映射: {model_name}[{class_id}] -> '{custom_name}'")
            return True
        except Exception as e:
            self.logger.error(f"添加映射失败: {e}")
            return False
    
    def remove_mapping(self, model_name: str, class_id: int) -> bool:
        """
        删除名称映射
        
        Args:
            model_name: 模型名称
            class_id: 类别ID
            
        Returns:
            是否成功删除
        """
        try:
            if model_name in self.mappings:
                class_id_str = str(class_id)
                if class_id_str in self.mappings[model_name]:
                    del self.mappings[model_name][class_id_str]
                    self.logger.info(f"删除映射: {model_name}[{class_id}]")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"删除映射失败: {e}")
            return False
    
    def get_all_mappings(self) -> Dict:
        """获取所有映射配置"""
        return self.mappings.copy()
    
    def get_model_mappings(self, model_name: str) -> Dict:
        """获取指定模型的映射配置"""
        return self.mappings.get(model_name, {}).copy()

    def import_config_mappings(self) -> bool:
        """
        导入config目录中的映射配置

        Returns:
            是否成功导入
        """
        try:
            project_root = Path(__file__).parent.parent
            config_mapping_file = project_root / "config" / "product_name_mapping.json"

            if not config_mapping_file.exists():
                self.logger.warning(f"config目录中的映射文件不存在: {config_mapping_file}")
                return False

            with open(config_mapping_file, 'r', encoding='utf-8') as f:
                config_mappings = json.load(f)

            # 合并映射配置
            for model_name, mappings in config_mappings.items():
                if model_name not in self.mappings:
                    self.mappings[model_name] = {}

                # 合并映射，config中的映射会覆盖现有的
                old_count = len(self.mappings[model_name])
                self.mappings[model_name].update(mappings)
                new_count = len(self.mappings[model_name])

                self.logger.info(f"模型 {model_name}: 导入 {len(mappings)} 个映射，总计 {new_count} 个映射")

            # 保存合并后的映射
            if self.save_mappings():
                self.logger.info(f"成功导入并保存config目录中的映射配置")
                return True
            else:
                self.logger.error("导入成功但保存失败")
                return False

        except Exception as e:
            self.logger.error(f"导入config映射配置失败: {e}")
            return False
    
    def apply_mappings_to_detections(self, detections: list) -> list:
        """
        对检测结果应用名称映射
        
        Args:
            detections: 检测结果列表
            
        Returns:
            应用映射后的检测结果列表
        """
        mapped_detections = []
        
        for detection in detections:
            # 复制检测结果
            mapped_detection = detection.copy()
            
            # 获取模型名称和类别信息
            model_name = detection.get('model', '')
            class_id = detection.get('class_id', -1)
            original_name = detection.get('class_name', '')
            
            # 应用名称映射
            if model_name and class_id >= 0:
                mapped_name = self.map_class_name(model_name, class_id, original_name)
                mapped_detection['class_name'] = mapped_name
                mapped_detection['original_class_name'] = original_name  # 保留原始名称
            
            mapped_detections.append(mapped_detection)
        
        return mapped_detections

# 全局名称映射器实例
_global_mapper = None

def get_global_mapper() -> ProductNameMapper:
    """获取全局名称映射器实例"""
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = ProductNameMapper()
    return _global_mapper

def apply_name_mappings(detections: list) -> list:
    """
    便捷函数：对检测结果应用名称映射
    
    Args:
        detections: 检测结果列表
        
    Returns:
        应用映射后的检测结果列表
    """
    mapper = get_global_mapper()
    return mapper.apply_mappings_to_detections(detections)