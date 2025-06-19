# -*- coding: utf-8 -*-
"""
智能商品识别系统配置文件
Configuration file for Smart Product Recognition System
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型文件路径配置
MODEL_PATHS = {
    'product_detector': os.path.join(PROJECT_ROOT, 'weight', 'product_detector_best.pt'),
    'region_detector': os.path.join(PROJECT_ROOT, 'weight', 'region_detector_best.pt'),
    'text_detector': os.path.join(PROJECT_ROOT, 'weight', 'cocotext_best.pt')
}

# 检测参数配置
DETECTION_CONFIG = {
    'confidence_threshold': 0.4,  # 降低阈值以便更容易检测到物体
    'iou_threshold': 0.45,
    'max_det': 1000,
    'device': 'cpu',  # 强制使用CPU避免GPU配置问题
    'verbose': False,  # 关闭详细输出
    'half': False,     # 不使用半精度
    'imgsz': 640       # 图像尺寸
}

# 图像处理参数
IMAGE_PROCESSING_CONFIG = {
    'resize_width': 640,
    'resize_height': 640,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'gaussian_blur_kernel': (5, 5),
    'morphology_kernel_size': (3, 3),
    'adaptive_threshold_max_value': 255,
    'adaptive_threshold_block_size': 11,
    'adaptive_threshold_c': 2
}

# OCR配置
OCR_CONFIG = {
    'tesseract_config': '--oem 3 --psm 6',
    'languages': 'chi_sim+eng',  # 中文优先，然后英文
    'confidence_threshold': 50,  # 降低阈值以提高中文识别率
    'chinese_support': True  # 启用中文支持
}

# 条形码配置
BARCODE_CONFIG = {
    'supported_formats': ['CODE128', 'EAN13', 'EAN8', 'UPC_A', 'UPC_E', 'CODE39'],
    'preprocessing_enabled': True
}

# GUI配置
GUI_CONFIG = {
    'window_title': '智能商品识别系统',
    'window_size': '1200x800',
    'theme': 'default',
    'font_family': 'Microsoft YaHei',
    'font_size': 10
}

# 输出配置
OUTPUT_CONFIG = {
    'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    'temp_dir': os.path.join(PROJECT_ROOT, 'temp'),
    'log_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'save_processed_images': True,
    'save_detection_results': True
}

# 应用场景配置
SCENARIO_CONFIG = {
    'personal_shopping': {
        'enabled': True,
        'extract_nutrition': True,
        'extract_barcode': True
    },
    'shelf_audit': {
        'enabled': True,
        'batch_processing': True,
        'generate_report': True
    }
}

# 创建必要的目录
def create_directories():
    """创建项目所需的目录"""
    dirs_to_create = [
        OUTPUT_CONFIG['results_dir'],
        OUTPUT_CONFIG['temp_dir'],
        OUTPUT_CONFIG['log_dir']
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

# 验证模型文件是否存在
def validate_models():
    """验证所有模型文件是否存在"""
    missing_models = []
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("警告：以下模型文件不存在：")
        for model in missing_models:
            print(f"  - {model}")
        return False
    return True

if __name__ == "__main__":
    create_directories()
    validate_models()
