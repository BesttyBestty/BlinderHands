#!/usr/bin/env python3
"""列出所有可用摄像头，并给每个摄像头拍一张，方便确认哪个对准了机械臂（HAND_CAMERA_INDEX）、哪个在看桌面（SCENE_CAMERA_INDEX）。"""

import cv2


def main(max_index: int = 6) -> None:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            path = f"camera_{i}.jpg"
            cv2.imwrite(path, frame)
            h, w = frame.shape[:2]
            print(f"索引 {i}：可用 {w}×{h}，亮度 {frame.mean():.0f}，已保存 {path}")
            found.append(i)
        else:
            print(f"索引 {i}：不可用")
    print(f"\n可用摄像头：{found}。打开 camera_*.jpg 看一眼：拍到机械臂手里物体的是手眼相机，拍到整个桌面的是全局相机。")
    print("注意：笔记本自带的前置摄像头通常是索引 0，别把它当成手眼相机。")


if __name__ == "__main__":
    main()
