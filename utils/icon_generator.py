# -*- coding: utf-8 -*-
"""
图标生成器
Icon Generator for Smart Product Recognition System
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class IconGenerator:
    """GUI图标生成器"""
    
    def __init__(self):
        self.icon_dir = "assets/icons"
        os.makedirs(self.icon_dir, exist_ok=True)
    
    def create_app_icon(self, size: int = 64, save_path: str = None) -> str:
        """
        创建应用程序图标
        
        Args:
            size: 图标尺寸
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            图标文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.icon_dir, f"app_icon_{size}x{size}.png")
        
        # 创建图像
        image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        
        # 背景渐变色
        for i in range(size):
            for j in range(size):
                r = int(64 + (i / size) * 100)
                g = int(128 + (j / size) * 100)
                b = int(255 - (i / size) * 50)
                alpha = 255
                image.putpixel((i, j), (r, g, b, alpha))
        
        # 绘制圆形背景
        margin = size // 8
        draw.ellipse([margin, margin, size-margin, size-margin], 
                    fill=(70, 130, 180, 255), outline=(50, 100, 150, 255), width=2)
        
        # 绘制商品图标（简化的购物车）
        center_x, center_y = size // 2, size // 2
        cart_size = size // 3
        
        # 购物车底部
        cart_bottom = center_y + cart_size // 4
        draw.rectangle([center_x - cart_size//2, cart_bottom - cart_size//4,
                       center_x + cart_size//2, cart_bottom], 
                      fill=(255, 255, 255, 255))
        
        # 购物车把手
        handle_start = center_x - cart_size//2 - cart_size//6
        draw.arc([handle_start, center_y - cart_size//2,
                 handle_start + cart_size//3, center_y + cart_size//2],
                start=270, end=90, fill=(255, 255, 255, 255), width=3)
        
        # 购物车轮子
        wheel_radius = cart_size // 8
        wheel_y = cart_bottom + wheel_radius
        # 左轮
        draw.ellipse([center_x - cart_size//3 - wheel_radius, wheel_y - wheel_radius,
                     center_x - cart_size//3 + wheel_radius, wheel_y + wheel_radius],
                    fill=(255, 255, 255, 255))
        # 右轮
        draw.ellipse([center_x + cart_size//6 - wheel_radius, wheel_y - wheel_radius,
                     center_x + cart_size//6 + wheel_radius, wheel_y + wheel_radius],
                    fill=(255, 255, 255, 255))
        
        # 保存图标
        image.save(save_path, 'PNG')
        return save_path
    
    def create_status_icon(self, size: int = 16, save_path: str = None) -> str:
        """
        创建状态栏小图标
        
        Args:
            size: 图标尺寸
            save_path: 保存路径
            
        Returns:
            图标文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.icon_dir, f"status_icon_{size}x{size}.png")
        
        # 创建图像
        image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        
        # 绘制简单的识别图标（眼睛形状）
        center_x, center_y = size // 2, size // 2
        
        # 外眼框
        draw.ellipse([2, center_y - size//4, size-2, center_y + size//4],
                    fill=(70, 130, 180, 255), outline=(50, 100, 150, 255))
        
        # 瞳孔
        pupil_size = size // 4
        draw.ellipse([center_x - pupil_size//2, center_y - pupil_size//2,
                     center_x + pupil_size//2, center_y + pupil_size//2],
                    fill=(255, 255, 255, 255))
        
        # 保存图标
        image.save(save_path, 'PNG')
        return save_path
    
    def create_ico_file(self, png_path: str, ico_path: str = None) -> str:
        """
        将PNG文件转换为ICO文件（用于Windows任务栏）
        
        Args:
            png_path: PNG文件路径
            ico_path: ICO文件保存路径
            
        Returns:
            ICO文件路径
        """
        if ico_path is None:
            ico_path = png_path.replace('.png', '.ico')
        
        try:
            # 打开PNG图像
            image = Image.open(png_path)
            
            # 创建多尺寸的ICO文件
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
            images = []
            
            for size in sizes:
                resized = image.resize(size, Image.Resampling.LANCZOS)
                images.append(resized)
            
            # 保存为ICO文件
            images[0].save(ico_path, format='ICO', sizes=sizes)
            return ico_path
            
        except Exception as e:
            print(f"创建ICO文件失败: {e}")
            return png_path

def generate_all_icons():
    """生成所有需要的图标"""
    generator = IconGenerator()
    
    # 生成应用图标
    app_icon_64 = generator.create_app_icon(64)
    app_icon_32 = generator.create_app_icon(32)
    app_icon_16 = generator.create_app_icon(16)
    
    # 生成状态栏图标
    status_icon = generator.create_status_icon(16)
    
    # 生成ICO文件（用于Windows任务栏）
    ico_file = generator.create_ico_file(app_icon_32)
    
    return {
        'app_icon_64': app_icon_64,
        'app_icon_32': app_icon_32,
        'app_icon_16': app_icon_16,
        'status_icon': status_icon,
        'ico_file': ico_file
    }

if __name__ == "__main__":
    icons = generate_all_icons()
    print("图标生成完成:")
    for name, path in icons.items():
        print(f"  {name}: {path}")
