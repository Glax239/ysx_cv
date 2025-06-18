# 智能商品识别系统 (Smart Product Recognition System)

一个基于深度学习和计算机视觉技术的智能商品识别系统，能够自动识别商品并从包装上精准定位、提取和解析多种关键信息。

## 🌟 项目特色

### 核心技术栈
- **多阶段YOLOv8模型**: 用于商品初定位和关键信息区域精定位
- **OpenCV数字图像处理**: 透视变换、自适应阈值、形态学操作、图像去噪与增强
- **Pyzbar**: 条形码解码
- **Pytesseract/PaddleOCR**: 文本信息识别

### 技术亮点
- **技术深度融合**: 完美结合前沿深度学习(YOLO)和经典计算机视觉(OpenCV)
- **创新性强**: 采用"层级检测"+"图像净化"的流水线设计
- **应用场景广泛**: 智慧零售、健康应用、辅助功能、市场研究

## 🎯 应用场景

### 场景A: 个人智能购物与健康助手
- **商品初检**: 实时检测摄像头画面中的商品
- **关键信息区域精定位**: 定位条形码和营养成分表区域
- **条形码解码**: 图像增强 → 去噪锐化 → 高精度解码
- **营养成分表解析**: 透视校正 → 二值化处理 → 形态学操作 → OCR识别

### 场景B: 自动化货架审计与分析
- **全货架扫描**: 批量检测货架上所有商品
- **品牌Logo识别**: 图像标准化 → 特征匹配/OCR识别
- **数据分析报告**: 品牌统计、缺货警告、错位提醒

## 📁 项目结构

```
cv_test/
├── main.py                 # 主程序入口
├── config.py              # 配置文件
├── requirements.txt       # 依赖包列表
├── install.py            # 安装脚本
├── README.md             # 项目说明
├── weight/               # 模型文件目录
│   ├── product_detector_best.pt
│   ├── region_detector_best.pt
│   ├── selected_brands_best.pt
│   └── cocotext_best.pt
├── core/                 # 核心模块
│   ├── __init__.py
│   ├── detection_engine.py      # 多阶段YOLO检测引擎
│   ├── image_processor.py       # 图像处理模块
│   └── information_extractor.py # 信息提取模块
├── gui/                  # GUI界面
│   ├── __init__.py
│   └── main_window.py           # 主窗口
├── utils/                # 工具模块
│   ├── __init__.py
│   ├── logger.py               # 日志配置
│   ├── file_utils.py           # 文件管理
│   └── scenario_processor.py   # 场景处理器
├── results/              # 结果输出目录
├── temp/                 # 临时文件目录
└── logs/                 # 日志文件目录
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Conda环境: `smart-product-analysis`
- 操作系统: Windows/Linux/macOS

### 安装步骤

1. **激活conda环境**
   ```bash
   conda activate smart-product-analysis
   ```

2. **运行安装脚本**
   ```bash
   python install.py
   ```

3. **手动安装Tesseract OCR** (如果需要)
   - Windows: 从 [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) 下载安装
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

### 使用方法

#### GUI模式 (推荐)
```bash
python main.py
```

#### 命令行模式
```bash
# 基础检测
python main.py --cli --input image.jpg

# 个人购物助手场景
python main.py --cli --input image.jpg --scenario personal_shopping

# 货架审计场景
python main.py --cli --input image.jpg --scenario shelf_audit

# 指定输出文件
python main.py --cli --input image.jpg --output result.json

# 显示详细日志
python main.py --cli --input image.jpg --verbose
```

## 🔧 配置说明

### 模型配置
在 `config.py` 中可以调整：
- 检测置信度阈值
- IoU阈值
- 设备选择 (CPU/GPU)

### 图像处理参数
- CLAHE参数
- 高斯模糊核大小
- 形态学操作核大小
- 自适应阈值参数

### OCR配置
- Tesseract配置参数
- 支持语言设置
- 置信度阈值

## 📊 功能特性

### 检测功能
- ✅ 商品检测
- ✅ 关键信息区域检测 (条形码、营养表等)
- ✅ 品牌Logo检测
- ✅ 文本区域检测

### 信息提取
- ✅ 条形码解码 (支持多种格式)
- ✅ 营养成分表解析
- ✅ 品牌识别 (OCR + 特征匹配)
- ✅ 通用文本识别

### 图像处理
- ✅ 透视校正
- ✅ 图像增强 (CLAHE、锐化)
- ✅ 去噪处理
- ✅ 形态学操作
- ✅ 自适应阈值

### 应用场景
- ✅ 个人购物助手
- ✅ 货架审计分析
- ✅ 批量处理
- ✅ 结果导出

## 🎨 GUI界面功能

### 主要功能
- 图像加载与显示
- 实时检测与结果展示
- 多标签页结果查看
- 图像缩放与导航
- 结果保存与导出

### 界面布局
- **工具栏**: 文件操作、检测控制、场景选择
- **图像面板**: 图像显示、缩放控制
- **结果面板**: 检测结果树、信息提取、统计分析
- **状态栏**: 实时状态显示

## 📈 性能优化

### 检测性能
- 多线程处理
- 批量检测优化
- 内存管理优化

### 图像处理
- 自适应参数调整
- 多尺度处理
- 并行处理支持

## 🔍 故障排除

### 常见问题

1. **模型文件缺失**
   - 确保 `weight/` 目录下有所有 `.pt` 文件
   - 检查文件路径配置

2. **OCR识别效果差**
   - 检查Tesseract安装
   - 调整图像预处理参数
   - 尝试不同的OCR引擎

3. **检测置信度低**
   - 调整置信度阈值
   - 改善图像质量
   - 检查光照条件

4. **内存不足**
   - 减小图像尺寸
   - 调整批处理大小
   - 使用CPU模式

## 📝 开发说明

### 扩展新功能
1. 在 `core/` 目录下添加新模块
2. 在 `config.py` 中添加相关配置
3. 更新 `requirements.txt` 添加新依赖
4. 在GUI中添加对应界面

### 添加新的检测模型
1. 将模型文件放入 `weight/` 目录
2. 在 `config.py` 中添加模型路径
3. 在 `DetectionEngine` 中添加加载逻辑

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**智能商品识别系统** - 让商品信息提取变得简单高效！
