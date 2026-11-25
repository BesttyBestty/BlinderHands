"""检测当前系统里有几个可用摄像头，用于确认 CAMERA_INDEX 该填几"""

import cv2


def check_cameras(max_index=6):
    print("开始检测可用摄像头...")
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                brightness = frame.mean()
                print(f"索引 {i}: 可用, 画面亮度 {brightness:.2f}")
                available.append(i)
            else:
                print(f"索引 {i}: 能打开但读取失败")
        else:
            print(f"索引 {i}: 不可用")
        cap.release()

    print(f"\n共检测到 {len(available)} 个可用摄像头, 索引分别是: {available}")
    return available


if __name__ == "__main__":
    check_cameras()
