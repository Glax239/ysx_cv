# 智能商品识别与健康分析系统 - 详细部署文档

## 📋 目录

1. [系统要求](#系统要求)
2. [环境准备](#环境准备)
3. [项目下载与安装](#项目下载与安装)
4. [依赖库安装](#依赖库安装)
5. [模型文件配置](#模型文件配置)
6. [配置文件设置](#配置文件设置)
7. [启动与运行](#启动与运行)
8. [功能验证](#功能验证)
9. [性能优化](#性能优化)
10. [故障排除](#故障排除)
11. [生产环境部署](#生产环境部署)
12. [维护与更新](#维护与更新)

---

## 🖥️ 系统要求

### 本机配置（可参考）
- **操作系统**: Windows 11
- **Python版本**: Python 3.8 - 3.11（本机3.9.3）
- **内存**: 40GB RAM
- **存储空间**: 1T
- **处理器**: i7-13650HX

---

## 🔧 环境准备

### 1. Python环境安装

#### Windows系统
```powershell
# 下载并安装Python 3.9 (推荐版本)
# 访问 https://www.python.org/downloads/windows/
# 下载Python 3.9.x版本，安装时勾选"Add Python to PATH"

# 验证安装
python --version
pip --version
```

#### Ubuntu/Debian系统（仅供参考）
```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装Python 3.9和相关工具
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev -y

# 安装系统依赖
sudo apt install build-essential cmake pkg-config -y
sudo apt install libjpeg-dev libtiff5-dev libpng-dev -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev -y
sudo apt install libgtk2.0-dev libcanberra-gtk-module -y
sudo apt install libxvidcore-dev libx264-dev -y
sudo apt install libgl1-mesa-glx libglib2.0-0 -y

# 验证安装
python3.9 --version
pip3 --version
```

#### macOS系统（仅供参考）
```bash
# 使用Homebrew安装Python
brew install python@3.9

# 安装系统依赖
brew install cmake pkg-config
brew install jpeg libpng libtiff openexr
brew install eigen tbb

# 验证安装
python3 --version
pip3 --version
```

### 2. CUDA环境配置 (GPU用户)

#### CUDA Toolkit安装
```bash
# 检查GPU信息
nvidia-smi

# Ubuntu安装CUDA 11.8 (推荐)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证CUDA安装
nvcc --version
```

#### Windows CUDA安装
1. 访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
2. 选择Windows x86_64版本
3. 下载并安装CUDA Toolkit 11.8
4. 重启计算机
5. 验证安装: `nvcc --version`

---

## 📦 项目下载与安装

### 1. 克隆项目
```bash
# 使用Git克隆项目
git clone https://github.com/Glax239/ysx_cv.git

# 或者下载ZIP文件并解压
```

### 2. 创建虚拟环境 (使用Anaconda)
```bash
# 创建Anaconda虚拟环境
conda create -n cv_test python=3.9 -y

# 激活虚拟环境
# Windows:
conda activate cv_test

# Linux/macOS:
conda activate cv_test

# 验证虚拟环境
conda info --envs  # 查看所有环境
python --version   # 验证Python版本
which python       # Linux/macOS - 查看Python路径
where python       # Windows - 查看Python路径

# 如果需要删除环境
# conda remove -n cv_test --all
```

### 3. 升级pip和基础工具
```bash
# 升级pip到最新版本
python -m pip install --upgrade pip

# 安装基础构建工具
pip install wheel setuptools
```

---

## 📚 依赖库安装

### 1. 核心依赖安装
```bash
# 确保已激活conda环境
conda activate cv_test

# 优先使用conda安装科学计算库 (推荐)
conda install numpy pandas matplotlib scikit-image opencv -y

# 安装剩余依赖
pip install -r requirements.txt

# 如果网络较慢，使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者完全使用pip安装
# pip install -r requirements.txt
```

### 2. 分步骤安装 (可选)
```bash
# 确保已激活conda环境
conda activate cv_test

# 1. 深度学习框架
# 使用conda安装PyTorch (推荐)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# 或使用pip安装
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# 2. 计算机视觉库
pip install opencv-python==4.8.0.74
pip install Pillow==9.5.0
pip install scikit-image==0.21.0

# 3. YOLO相关 (YOLOv8)
pip install ultralytics>=8.0.0

# 4. OCR库
pip install paddlepaddle==2.5.1
pip install paddleocr==2.7.0.3
pip install easyocr==1.7.0

# 5. 条形码识别
pip install pyzbar==0.1.9
pip install python-barcode==0.14.0

# 6. GUI框架
pip install PyQt5==5.15.9
pip install PyQt5-tools==5.15.9.3.3

# 7. 数据处理
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.1

# 8. 其他工具库
pip install requests==2.31.0
pip install tqdm==4.65.0
pip install psutil==5.9.5
```

### 3. 验证安装
```bash
# 确保已激活conda环境
conda activate cv_test
```

```python
# 创建测试脚本 test_installation.py
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import paddleocr
from PyQt5.QtWidgets import QApplication
import sys

print("=== 依赖库版本检查 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"OpenCV版本: {cv2.__version__}")
print(f"NumPy版本: {np.__version__}")

# 检查CUDA支持
if torch.cuda.is_available():
    print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("CUDA不可用，将使用CPU模式")

# 检查GUI支持
try:
    app = QApplication([])
    print("PyQt5 GUI支持正常")
    app.quit()
except Exception as e:
    print(f"GUI支持异常: {e}")

print("\n=== 安装验证完成 ===")
```

```bash
# 运行验证脚本
python test_installation.py
```

---

## 🎯 模型文件配置

### 1. 创建模型目录结构
```bash
# 创建必要的目录
mkdir -p models/yolo
mkdir -p models/weights
mkdir -p data/test_images
mkdir -p output/results
mkdir -p output/processed
mkdir -p logs
```

### 2. 下载预训练模型
```bash
# 下载YOLOv8模型 (自动下载)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 3. 配置自定义模型 (如果有)
```bash
# 将自定义训练的模型文件放置到指定目录
cp your_custom_model.pt models/weights/

# 更新模型配置文件
# 编辑 config.py 中的模型路径
```

### 4. 验证模型加载
```python
# 创建模型验证脚本 test_models.py
from ultralytics import YOLO
import torch

print("=== 模型加载测试 ===")

# 测试YOLOv8模型
try:
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8n模型加载成功")
except Exception as e:
    print(f"❌ YOLOv8n模型加载失败: {e}")

# 测试自定义模型 (如果存在)
try:
    custom_model = YOLO('models/weights/your_custom_model.pt')
    print("✅ 自定义模型加载成功")
except Exception as e:
    print(f"⚠️ 自定义模型加载失败: {e}")

print("\n=== 模型测试完成 ===")
```

---

## ⚙️ 配置文件设置

### 1. 基础配置检查
```python
# 检查config.py文件
cat config.py
```

### 2. 路径配置调整
```python
# 编辑config.py文件
# 确保以下路径配置正确

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 模型文件路径
YOLO_MODELS = {
    'product': os.path.join(MODELS_DIR, 'weights', 'product_detection.pt'),
    'region': os.path.join(MODELS_DIR, 'weights', 'region_detection.pt'),
    'text': os.path.join(MODELS_DIR, 'weights', 'text_detection.pt')
}

# 如果没有自定义模型，使用预训练模型
if not os.path.exists(YOLO_MODELS['product']):
    YOLO_MODELS['product'] = 'yolov8n.pt'
if not os.path.exists(YOLO_MODELS['region']):
    YOLO_MODELS['region'] = 'yolov8s.pt'
if not os.path.exists(YOLO_MODELS['text']):
    YOLO_MODELS['text'] = 'yolov8n.pt'
```

### 3. 性能配置优化
```python
# 根据硬件配置调整参数

# GPU配置
if torch.cuda.is_available():
    DEVICE = 'cuda'
    BATCH_SIZE = 4  # 根据显存调整
    NUM_WORKERS = 4
else:
    DEVICE = 'cpu'
    BATCH_SIZE = 1
    NUM_WORKERS = 2

# 检测参数
DETECTION_CONFIG = {
    'confidence_threshold': 0.25,  # 置信度阈值
    'iou_threshold': 0.45,         # NMS阈值
    'max_detections': 100,         # 最大检测数量
    'image_size': 640              # 输入图像尺寸
}

# OCR配置
OCR_CONFIG = {
    'use_gpu': torch.cuda.is_available(),
    'lang': 'ch',                  # 中文识别
    'det_algorithm': 'DB',         # 检测算法
    'rec_algorithm': 'CRNN'        # 识别算法
}
```

### 4. 创建配置验证脚本
```python
# 创建 verify_config.py
import os
import torch
from config import *

def verify_configuration():
    print("=== 配置验证 ===")
    
    # 检查目录
    directories = [MODELS_DIR, DATA_DIR, OUTPUT_DIR]
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ 目录存在: {directory}")
        else:
            print(f"❌ 目录不存在: {directory}")
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 已创建目录: {directory}")
    
    # 检查模型文件
    for model_name, model_path in YOLO_MODELS.items():
        if os.path.exists(model_path) or model_path.endswith('.pt'):
            print(f"✅ {model_name}模型配置正确: {model_path}")
        else:
            print(f"⚠️ {model_name}模型文件不存在: {model_path}")
    
    # 检查设备配置
    print(f"✅ 计算设备: {DEVICE}")
    print(f"✅ 批处理大小: {BATCH_SIZE}")
    
    print("\n=== 配置验证完成 ===")

if __name__ == "__main__":
    verify_configuration()
```

```bash
# 运行配置验证
python verify_config.py
```

---

## 🚀 启动与运行

### 1. GUI模式启动
```bash
# 启动PyQt5图形界面
python start_pyqt5_gui.py

# 如果遇到显示问题 (Linux)
export QT_QPA_PLATFORM=xcb
python start_pyqt5_gui.py

# macOS可能需要
export QT_QPA_PLATFORM=cocoa
python start_pyqt5_gui.py
```

### 2. 命令行模式
```bash
# 运行示例程序
python example_usage.py

# 处理单张图片
python -c "
from core.simple_information_extractor import SimpleInformationExtractor
extractor = SimpleInformationExtractor()
result = extractor.extract_comprehensive_info('data/test_images/sample.jpg')
print(result)
"

# 批量处理图片
python -c "
import os
from core.simple_information_extractor import SimpleInformationExtractor

extractor = SimpleInformationExtractor()
test_dir = 'data/test_images'
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_dir, filename)
        result = extractor.extract_comprehensive_info(image_path)
        print(f'{filename}: {result}')
"
```
### 然后就可以正常运行进行图片检测啦！