"""
物体追踪
"""
import cv2

from ultralytics import YOLO


def main():
    """"""
    model = YOLO(model="../../../models/yolov8n.pt")

    cap = cv2.VideoCapture("./video2.mp4")

    while cap.isOpened():
        status, frame = cap.read()
        if status:
            results = model.track(source=frame, persist=True)

            img = results[0].plot()
            cv2.imshow(winname="demo", mat=img)

            if cv2.waitKey(delay=100) == 27:
                break
        else:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
