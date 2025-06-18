# -*- coding: utf-8 -*-
"""
图像处理模块
Image Processing Module with OpenCV
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from skimage import measure, morphology
from scipy import ndimage

from config import IMAGE_PROCESSING_CONFIG

class ImageProcessor:
    """高级图像处理类，专注于商品信息提取的图像优化"""
    
    def __init__(self):
        """初始化图像处理器"""
        self.logger = logging.getLogger(__name__)
        self.config = IMAGE_PROCESSING_CONFIG
    
    def enhance_barcode_region(self, image: np.ndarray) -> np.ndarray:
        """
        条形码区域增强处理
        
        Args:
            image: 输入的条形码区域图像
            
        Returns:
            增强后的条形码图像
        """
        try:
            # 1. 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 2. 应用CLAHE增强对比度
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            enhanced = clahe.apply(gray)
            
            # 3. 高斯模糊去噪
            blurred = cv2.GaussianBlur(
                enhanced, 
                self.config['gaussian_blur_kernel'], 
                0
            )
            
            # 4. 拉普拉斯锐化
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            sharpened = np.uint8(np.absolute(laplacian))
            
            # 5. 组合原图和锐化结果
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            # 6. 自适应阈值处理
            binary = cv2.adaptiveThreshold(
                result,
                self.config['adaptive_threshold_max_value'],
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config['adaptive_threshold_block_size'],
                self.config['adaptive_threshold_c']
            )
            
            self.logger.info("条形码区域增强处理完成")
            return binary
            
        except Exception as e:
            self.logger.error(f"条形码增强处理失败: {e}")
            return image
    
    def correct_perspective(self, image: np.ndarray, corners: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        透视校正，将倾斜的表格或文本区域校正为正视图
        
        Args:
            image: 输入图像
            corners: 四个角点坐标，如果为None则自动检测
            
        Returns:
            透视校正后的图像
        """
        try:
            if corners is None:
                corners = self._detect_corners(image)
            
            if len(corners) != 4:
                self.logger.warning("未能检测到四个角点，跳过透视校正")
                return image
            
            # 排序角点：左上、右上、右下、左下
            corners = self._order_corners(corners)
            
            # 计算目标矩形的尺寸
            width = max(
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[3])
            )
            height = max(
                np.linalg.norm(corners[3] - corners[0]),
                np.linalg.norm(corners[2] - corners[1])
            )
            
            # 定义目标角点
            dst_corners = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            src_corners = np.array(corners, dtype=np.float32)
            transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
            
            # 应用透视变换
            corrected = cv2.warpPerspective(
                image, 
                transform_matrix, 
                (int(width), int(height))
            )
            
            self.logger.info("透视校正完成")
            return corrected
            
        except Exception as e:
            self.logger.error(f"透视校正失败: {e}")
            return image
    
    def _detect_corners(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """自动检测图像中的四个角点"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 找到最大的四边形轮廓
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                # 轮廓近似
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    return [tuple(point[0]) for point in approx]
            
            return []
            
        except Exception as e:
            self.logger.error(f"角点检测失败: {e}")
            return []
    
    def _order_corners(self, corners: List[Tuple[int, int]]) -> np.ndarray:
        """按照左上、右上、右下、左下的顺序排列角点"""
        corners = np.array(corners)
        
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 按角度排序
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # 重新排列，确保从左上角开始
        ordered = corners[sorted_indices]
        
        # 找到左上角（x+y最小的点）
        sums = ordered[:, 0] + ordered[:, 1]
        top_left_idx = np.argmin(sums)
        
        # 重新排列，使左上角在第一位
        ordered = np.roll(ordered, -top_left_idx, axis=0)
        
        return ordered
    
    def enhance_text_region(self, image: np.ndarray) -> np.ndarray:
        """
        文本区域增强处理，优化OCR识别效果
        
        Args:
            image: 输入的文本区域图像
            
        Returns:
            增强后的文本图像
        """
        try:
            # 1. 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 2. 去噪处理
            denoised = cv2.medianBlur(gray, 3)
            
            # 3. 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # 4. 形态学操作去除噪声
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                self.config['morphology_kernel_size']
            )
            
            # 开运算去除小噪声
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 闭运算连接断裂的文字
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # 5. 如果是营养成分表，尝试去除表格线
            cleaned = self._remove_table_lines(closed)
            
            self.logger.info("文本区域增强处理完成")
            return cleaned
            
        except Exception as e:
            self.logger.error(f"文本区域增强处理失败: {e}")
            return image
    
    def _remove_table_lines(self, binary_image: np.ndarray) -> np.ndarray:
        """去除表格中的横竖线条，保留文字"""
        try:
            # 检测水平线
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
            
            # 检测垂直线
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)
            
            # 合并线条
            table_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # 从原图中减去线条
            result = cv2.subtract(binary_image, table_lines)
            
            return result
            
        except Exception as e:
            self.logger.error(f"去除表格线失败: {e}")
            return binary_image
    
    def enhance_logo_region(self, image: np.ndarray) -> np.ndarray:
        """
        Logo区域增强处理，为特征匹配做准备
        
        Args:
            image: 输入的Logo区域图像
            
        Returns:
            增强后的Logo图像
        """
        try:
            # 1. 尺寸标准化
            target_size = (128, 128)  # 标准化尺寸
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            # 2. 转换为灰度图
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized.copy()
            
            # 3. 直方图均衡化
            equalized = cv2.equalizeHist(gray)
            
            # 4. 高斯模糊去噪
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            
            # 5. 边缘增强
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            self.logger.info("Logo区域增强处理完成")
            return sharpened
            
        except Exception as e:
            self.logger.error(f"Logo区域增强处理失败: {e}")
            return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            
        Returns:
            旋转后的图像
        """
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 应用旋转
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            return rotated
            
        except Exception as e:
            self.logger.error(f"图像旋转失败: {e}")
            return image
    
    def detect_text_orientation(self, image: np.ndarray) -> float:
        """
        检测文本方向，返回需要旋转的角度
        
        Args:
            image: 输入的文本图像
            
        Returns:
            建议的旋转角度
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 霍夫直线检测
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    angles.append(angle)
                
                # 计算主要角度
                angles = np.array(angles)
                # 将角度归一化到[-45, 45]范围
                angles = angles % 180
                angles[angles > 90] -= 180
                
                # 使用直方图找到最常见的角度
                hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
                max_bin = np.argmax(hist)
                dominant_angle = (bins[max_bin] + bins[max_bin + 1]) / 2
                
                return -dominant_angle  # 返回校正角度
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"文本方向检测失败: {e}")
            return 0.0
    
    def crop_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                   padding: int = 5) -> np.ndarray:
        """
        裁剪指定区域，并添加边距
        
        Args:
            image: 输入图像
            bbox: 边界框 (x1, y1, x2, y2)
            padding: 边距像素
            
        Returns:
            裁剪后的图像
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # 添加边距并确保不超出图像边界
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            return image[y1:y2, x1:x2]
            
        except Exception as e:
            self.logger.error(f"区域裁剪失败: {e}")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            keep_aspect_ratio: 是否保持宽高比
            
        Returns:
            调整尺寸后的图像
        """
        try:
            if keep_aspect_ratio:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # 计算缩放比例
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # 调整尺寸
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 创建目标尺寸的画布并居中放置
                canvas = np.zeros((target_h, target_w, 3) if len(image.shape) == 3 else (target_h, target_w), 
                                dtype=image.dtype)
                
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                if len(image.shape) == 3:
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                else:
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return canvas
            else:
                return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                
        except Exception as e:
            self.logger.error(f"图像尺寸调整失败: {e}")
            return image
