"""
拍照 -> Qwen-VL 识别日期 -> 判断是否过期 -> 语音播报
运行前需要：
1. pip3 install opencv-python openai
2. 设置环境变量 DASHSCOPE_API_KEY（不要把 key 写死在代码里）
3. 确认摄像头索引（用 check_cameras.py 先检测）
"""

import os
import cv2
import time
import base64
import re
from datetime import datetime
from openai import OpenAI

CAMERA_INDEX = 0  # 根据实际检测结果调整


def parse_and_check(response_text: str):
    """从 Qwen-VL 返回的文本里解析出'有效期'并判断是否过期"""
    match = re.search(
        r'有效期[至到]?[:：]?\s*(\d{4})[-/年](\d{1,2})[-/月]?(\d{1,2})?',
        response_text
    )
    if not match:
        return "未能识别出有效期,建议旋转角度重试", None

    year, month = int(match.group(1)), int(match.group(2))
    day = match.group(3)

    if day:
        expiry_date = datetime(year, month, int(day))
    else:
        # 只有年月时，按下个月1号作为保守的过期判定线
        expiry_date = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    if expiry_date < datetime.now():
        return f"已过期,有效期止于{expiry_date.year}年{expiry_date.month}月,不建议使用", False
    else:
        return f"在保质期内,有效期到{expiry_date.year}年{expiry_date.month}月", True


def capture_and_check():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None

    # 预热，等待自动曝光稳定
    for _ in range(15):
        cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    if not ret:
        print("拍摄失败")
        cap.release()
        return None

    brightness = frame.mean()
    print(f"画面平均亮度: {brightness:.2f}")
    cv2.imwrite("drug_photo.jpg", frame)
    cap.release()

    with open("drug_photo.jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": (
                        "请找出图片中的生产日期和有效期至这两项信息,"
                        "严格按照下面格式回复,不要有多余文字:\n"
                        "生产日期:XXXX年XX月\n有效期至:XXXX年XX月\n"
                        "如果某一项找不到,对应位置写未找到。"
                    )}
                ]
            }
        ]
    )

    result_text = response.choices[0].message.content.strip()
    print(f"识别结果:\n{result_text}\n")

    final_text, is_valid = parse_and_check(result_text)
    print(f"最终结果: {final_text}")

    os.system(f'say "{final_text}"')  # macOS 自带 TTS，其他系统请替换

    return final_text


if __name__ == "__main__":
    capture_and_check()
