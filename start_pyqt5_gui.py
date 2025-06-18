# -*- coding: utf-8 -*-
"""
启动PyQt5版本的GUI
Launch PyQt5 version of the GUI
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函数"""
    try:
        # 导入PyQt5主窗口
        from gui.pyqt5_main_window import main as pyqt5_main
        
        print("启动PyQt5版本的智能商品识别系统...")
        print("=" * 50)
        print("系统特性:")
        print("✓ 现代化PyQt5界面")
        print("✓ 多线程检测处理")
        print("✓ 增强的OCR文本识别")
        print("✓ 完整的文本输出功能")
        print("✓ 专业的结果展示")
        print("=" * 50)
        
        # 启动PyQt5应用
        pyqt5_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装PyQt5:")
        print("pip install PyQt5")
        sys.exit(1)
        
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
