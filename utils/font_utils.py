# -*- coding: utf-8 -*-
"""
字体工具模块
Font Utilities Module for Chinese Text Rendering
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Tuple, Union
import logging

class ChineseFontRenderer:
    """中文字体渲染器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.font_paths = self._get_system_fonts()
        self.default_font = self._load_default_font()
        
    def _get_system_fonts(self) -> list:
        """获取系统中文字体路径"""
        font_paths = []
        
        # Windows系统字体路径
        windows_fonts = [
            r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
            r"C:\Windows\Fonts\msyhbd.ttc",    # 微软雅黑粗体
            r"C:\Windows\Fonts\simhei.ttf",    # 黑体
            r"C:\Windows\Fonts\simsun.ttc",    # 宋体
            r"C:\Windows\Fonts\simkai.ttf",    # 楷体
            r"C:\Windows\Fonts\simfang.ttf",   # 仿宋
        ]
        
        # 检查字体文件是否存在
        for font_path in windows_fonts:
            if os.path.exists(font_path):
                font_paths.append(font_path)
        
        # 如果没有找到系统字体，尝试其他路径
        if not font_paths:
            # 可以添加其他系统的字体路径
            other_paths = [
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            ]
            for font_path in other_paths:
                if os.path.exists(font_path):
                    font_paths.append(font_path)
        
        return font_paths
    
    def _load_default_font(self, size: int = 20):
        """加载默认字体"""
        try:
            if self.font_paths:
                # 优先使用微软雅黑
                preferred_fonts = [
                    r"C:\Windows\Fonts\msyh.ttc",
                    r"C:\Windows\Fonts\msyhbd.ttc"
                ]
                
                for font_path in preferred_fonts:
                    if font_path in self.font_paths:
                        return ImageFont.truetype(font_path, size)
                
                # 如果没有找到首选字体，使用第一个可用字体
                return ImageFont.truetype(self.font_paths[0], size)
            else:
                self.logger.warning("未找到系统中文字体，使用默认字体")
                return ImageFont.load_default()
        except Exception as e:
            self.logger.error(f"加载字体失败: {e}")
            return ImageFont.load_default()
    
    def draw_chinese_text(self, 
                         image: np.ndarray, 
                         text: str, 
                         position: Tuple[int, int], 
                         font_size: int = 20, 
                         color: Tuple[int, int, int] = (255, 255, 255),
                         background_color: Union[Tuple[int, int, int], None] = None,
                         background_padding: int = 5) -> np.ndarray:
        """
        在图像上绘制中文文本
        
        Args:
            image: 输入图像 (BGR格式)
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font_size: 字体大小
            color: 文本颜色 (RGB格式)
            background_color: 背景颜色 (RGB格式)，如果为None则不绘制背景
            background_padding: 背景内边距
            
        Returns:
            绘制文本后的图像
        """
        try:
            # 转换BGR到RGB
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 加载字体
            font = self._load_default_font(font_size)
            
            # 获取文本尺寸
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x, y = position
            
            # 绘制背景
            if background_color is not None:
                bg_x1 = x - background_padding
                bg_y1 = y - background_padding
                bg_x2 = x + text_width + background_padding
                bg_y2 = y + text_height + background_padding
                
                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=background_color)
            
            # 绘制文本
            draw.text((x, y), text, font=font, fill=color)
            
            # 转换回BGR格式
            result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return result_image
            
        except Exception as e:
            self.logger.error(f"绘制中文文本失败: {e}")
            # 回退到原始方法
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size/20, color, 2)
            return image
    
    def get_text_size(self, text: str, font_size: int = 20) -> Tuple[int, int]:
        """
        获取文本尺寸
        
        Args:
            text: 文本内容
            font_size: 字体大小
            
        Returns:
            文本宽度和高度 (width, height)
        """
        try:
            # 创建临时图像来测量文本尺寸
            temp_image = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(temp_image)
            font = self._load_default_font(font_size)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            return width, height
            
        except Exception as e:
            self.logger.error(f"获取文本尺寸失败: {e}")
            # 回退到估算方法
            return len(text) * font_size, font_size

# 全局字体渲染器实例
chinese_font_renderer = ChineseFontRenderer()

def draw_chinese_text_on_image(image: np.ndarray, 
                              text: str, 
                              position: Tuple[int, int], 
                              font_size: int = 20, 
                              color: Tuple[int, int, int] = (255, 255, 255),
                              background_color: Union[Tuple[int, int, int], None] = None) -> np.ndarray:
    """
    在图像上绘制中文文本的便捷函数
    """
    return chinese_font_renderer.draw_chinese_text(
        image, text, position, font_size, color, background_color
    )
