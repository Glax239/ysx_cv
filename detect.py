import cv2
import numpy as np
from ultralytics import YOLO
from utils.font_utils import draw_chinese_text_on_image

# --- 1. 配置 ---
# 加载你训练好的 YOLO 模型
MODEL_PATH = r'weight\product_detector_best.pt'  # 商品检测模型
# 你想要进行检测的图片路径
IMAGE_PATH = 'wafle.jpg'
# 设置置信度阈值
CONFIDENCE_THRESHOLD = 0.5
# 输出图片的保存路径
OUTPUT_IMAGE_PATH = 'output_image_colored.jpg'

# --- 2. 加载模型和图片 ---
try:
    # 加载 YOLO 模型
    model = YOLO(MODEL_PATH)
    # 读取图片
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"无法找到或打开图片: {IMAGE_PATH}")
except Exception as e:
    print(f"错误: {e}")
    exit()

# --- 3. 为类别动态生成颜色 ---
# 获取模型中所有的类别名称
class_names = model.names
# 使用 numpy 为每个类别生成一个唯一的颜色
# 我们设置一个随机种子，以确保每次运行代码时，相同类别的颜色都是固定的
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')


# --- 4. 进行目标检测 ---
# 使用模型对图片进行预测
results = model.predict(image, conf=CONFIDENCE_THRESHOLD)

# --- 5. 绘制结果 ---
for result in results:
    boxes = result.boxes
    for box in boxes:
        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 获取置信度
        confidence = float(box.conf[0])
        
        # 获取类别ID
        class_id = int(box.cls[0])
        
        # --- 优化点：根据类别ID获取专属颜色 ---
        class_name = class_names[class_id]
        color = colors[class_id].tolist() # 将颜色从numpy array转为list

        # 准备标签文本
        label = f'{class_name} {confidence:.2f}'
        
        # --- 优化点：增大边框和字体尺寸 ---
        # 1. 绘制更粗的边界框
        border_thickness = 3 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, border_thickness)
        
        # 2. 设置更清晰的字体样式
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 增大了字体大小
        font_thickness = 2 # 增大了字体粗细
        
        # 计算文本尺寸，用于绘制背景
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # 3. 绘制标签背景，确保文字清晰可读
        # 将背景放在边框的上方
        label_y1 = max(y1 - text_height - baseline, 0)
        label_y2 = y1
        cv2.rectangle(image, (x1, label_y1), (x1 + text_width, label_y2), color, -1)
          # 4. 使用中文字体渲染器绘制标签文本
        image = draw_chinese_text_on_image(
            image,
            label,
            (x1, max(y1 - 45, 0)),  # 增大文字位置偏移从30到45
            font_size=32,  # 增大字体大小从20到32
            color=(255, 255, 255),  # 白色文字
            background_color=tuple(color)  # 使用检测框同色背景
        )

# --- 6. 显示和保存图片 ---
cv2.imwrite(OUTPUT_IMAGE_PATH, image)
print(f"处理完成！优化后的结果已保存至: {OUTPUT_IMAGE_PATH}")

# 如果你想在运行时直接看到结果，可以取消下面代码的注释
cv2.imshow('YOLO Detection Result (Colored)', image)
cv2.waitKey(0)
# cv2.destroyAllWindows()