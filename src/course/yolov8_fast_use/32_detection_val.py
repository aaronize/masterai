"""
模型验证
"""
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(model="/Users/aaron/Workspace/masterai/runs/detect/train2/weights/best.pt")

    metrics = model.val()

    print(metrics.box)