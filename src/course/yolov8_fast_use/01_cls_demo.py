import cv2

from ultralytics import YOLO


def main():
    """
    使用预训练模型
    :return:
    """
    # 加载模型
    model = YOLO(model="../../../models/yolov8l-cls.pt")

    # 打开图像
    img = cv2.imread(filename="./iris.png")

    # 推理过程
    results = model.predict(source=img)

    # 绘制结果
    img = results[0].plot()
    cv2.imshow(winname="demo", mat=img)

    if cv2.waitKey(delay=30000) == 27:
        pass

    # 释放资源
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
