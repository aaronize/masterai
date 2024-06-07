"""
YOLOv8目标检测
"""
from ultralytics import YOLO


if __name__ == '__main__':
    # 构建模型
    model = YOLO(model="/Users/aaron/Workspace/masterai/src/course/yolov8_fast_use/cfg/yolov8n.yaml")

    # 训练模型
    model.train(data="/Users/aaron/Workspace/masterai/src/course/yolov8_fast_use/cfg/animals-dataset.yaml",
                epochs=10,
                imgsz=640)
