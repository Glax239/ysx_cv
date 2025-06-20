import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.restoration import denoise_bilateral

class EnhancedImageProcessor:
    """
    增强的图像预处理器，专门用于提高OCR识别准确率
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def enhance_for_ocr(self, image: np.ndarray, region_type: str = 'default') -> np.ndarray:
        """
        针对OCR优化的图像增强
        
        Args:
            image: 输入图像
            region_type: 区域类型 ('nutrition', 'text', 'default')
            
        Returns:
            增强后的图像
        """
        try:
            # 1. 基础预处理
            enhanced = self._basic_preprocessing(image)
            
            # 2. 根据区域类型选择特定处理
            if region_type == 'nutrition':
                enhanced = self._enhance_nutrition_table(enhanced)
            elif region_type == 'text':
                enhanced = self._enhance_text_region(enhanced)
            else:
                enhanced = self._enhance_general_text(enhanced)
            
            # 3. 最终优化
            enhanced = self._final_optimization(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {e}")
            return image
    
    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        基础预处理步骤
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 去噪
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _enhance_nutrition_table(self, image: np.ndarray) -> np.ndarray:
        """
        营养成分表专用增强
        """
        # 1. 去除表格线
        no_lines = self._remove_table_lines_advanced(image)
        
        # 2. 增强文字对比度
        enhanced = self._enhance_text_contrast(no_lines)
        
        # 3. 形态学操作优化字符
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def _enhance_text_region(self, image: np.ndarray) -> np.ndarray:
        """
        普通文本区域增强
        """
        # 1. 锐化
        sharpened = self._sharpen_image(image)
        
        # 2. 增强对比度
        enhanced = self._enhance_text_contrast(sharpened)
        
        # 3. 去除噪点
        cleaned = self._remove_noise(enhanced)
        
        return cleaned
    
    def _enhance_general_text(self, image: np.ndarray) -> np.ndarray:
        """
        通用文本增强
        """
        # 自适应阈值处理
        adaptive = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def _remove_table_lines_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        高级表格线去除算法
        """
        try:
            # 1. 检测水平线
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
            
            # 2. 检测垂直线
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
            
            # 3. 合并线条
            lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # 4. 从原图中减去线条
            result = cv2.subtract(image, lines)
            
            # 5. 进一步清理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"表格线去除失败: {e}")
            return image
    
    def _enhance_text_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强文字对比度
        """
        # 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Gamma校正
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lookup_table)
        
        return enhanced
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像锐化
        """
        # 拉普拉斯锐化
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image - 0.3 * laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        去除噪点
        """
        # 中值滤波去噪
        denoised = cv2.medianBlur(image, 3)
        
        # 形态学开运算去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _final_optimization(self, image: np.ndarray) -> np.ndarray:
        """
        最终优化处理
        """
        # 1. 二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 连通组件分析，去除过小的组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 计算平均组件大小
        areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
        if len(areas) > 0:
            mean_area = np.mean(areas)
            min_area = max(10, mean_area * 0.1)  # 最小面积阈值
            
            # 创建掩码，保留足够大的组件
            mask = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    mask[labels == i] = 255
            
            binary = mask
        
        return binary
    
    def adaptive_resize_for_ocr(self, image: np.ndarray, target_height: int = 64) -> np.ndarray:
        """
        自适应调整图像大小以优化OCR识别
        
        Args:
            image: 输入图像
            target_height: 目标高度
            
        Returns:
            调整大小后的图像
        """
        try:
            h, w = image.shape[:2]
            
            # 计算缩放比例
            if h < target_height:
                # 图像太小，需要放大
                scale = target_height / h
            elif h > target_height * 3:
                # 图像太大，需要缩小
                scale = (target_height * 2) / h
            else:
                # 大小合适，不需要调整
                return image
            
            # 计算新尺寸
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 使用高质量插值
            if scale > 1:
                # 放大使用立方插值
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                # 缩小使用区域插值
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return resized
            
        except Exception as e:
            self.logger.error(f"图像大小调整失败: {e}")
            return image
    
    def detect_text_orientation(self, image: np.ndarray) -> float:
        """
        检测文本方向
        
        Args:
            image: 输入图像
            
        Returns:
            旋转角度（度）
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    # 将角度标准化到[-45, 45]范围
                    if angle > 45:
                        angle = angle - 90
                    elif angle < -45:
                        angle = angle + 90
                    angles.append(angle)
                
                if angles:
                    # 返回最常见的角度
                    return np.median(angles)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"文本方向检测失败: {e}")
            return 0.0
    
    def correct_skew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        纠正图像倾斜
        
        Args:
            image: 输入图像
            angle: 旋转角度
            
        Returns:
            纠正后的图像
        """
        try:
            if abs(angle) < 0.5:  # 角度太小，不需要纠正
                return image
            
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 计算新的边界框大小
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_w = int((h * sin_angle) + (w * cos_angle))
            new_h = int((h * cos_angle) + (w * sin_angle))
            
            # 调整旋转矩阵的平移部分
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # 执行旋转
            rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            self.logger.error(f"倾斜纠正失败: {e}")
            return image
    
    def enhance_chinese_characters(self, image: np.ndarray) -> np.ndarray:
        """
        专门针对中文字符的增强处理
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        try:
            # 1. 增强对比度
            enhanced = self._enhance_text_contrast(image)
            
            # 2. 中文字符通常比较复杂，需要保持细节
            # 使用保边滤波
            filtered = cv2.bilateralFilter(enhanced, 9, 80, 80)
            
            # 3. 轻微的形态学操作，避免破坏字符结构
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            morphed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
            
            # 4. 自适应阈值，更好地处理不均匀光照
            binary = cv2.adaptiveThreshold(
                morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 3
            )
            
            return binary
            
        except Exception as e:
            self.logger.error(f"中文字符增强失败: {e}")
            return image