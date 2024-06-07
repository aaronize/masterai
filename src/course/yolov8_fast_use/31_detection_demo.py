"""
目标检测
"""
import cv2

from ultralytics import YOLO


def main():
    """"""
    # 加载模型
    model = YOLO(model="../../../models/yolov8n.pt")
    # model = YOLO(model="../../../runs/detect/train6/weights/best.pt")

    # 读取图像
    img = cv2.imread(filename="./cat01.jpeg")

    # 推理
    results = model.predict(source=img)

    # 结果绘制
    img = results[0].plot()

    # 显示图像
    cv2.imshow(winname="demo", mat=img)

    if cv2.waitKey(delay=30000) == 27:
        pass

    # 释放资源
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

