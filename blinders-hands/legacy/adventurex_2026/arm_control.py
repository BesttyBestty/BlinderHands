"""
松灵 NERO 机械臂 HTTP 接口控制
连接方式：有线网口（推荐，避免与外网 WiFi 冲突），默认 IP 10.90.0.150
Token 从机械臂 Web 后台登录后，用浏览器开发者工具抓包获取（约 8 小时过期）
"""

import os
import requests

ARM_BASE_URL = os.environ.get("ARM_BASE_URL", "http://10.90.0.150")
ARM_TOKEN = os.environ.get("ARM_TOKEN", "")


def move_arm_joints(joint_angles, token=None):
    """
    joint_angles: 长度为 7 的列表，弧度制，对应 J1~J7
    注意：请求体是纯数组，不要包 {"joints": [...]} 这种外层字段名
    """
    token = token or ARM_TOKEN
    url = f"{ARM_BASE_URL}/api/move/joint"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    response = requests.post(url, headers=headers, json=joint_angles, timeout=5)
    return response.json()


# 候选旋转角度，需要现场用 Web 端手动标定后替换成真实数值
CANDIDATE_ANGLES = [
    [-0.371598, 1.11352, 0.084282, -1.012291, -1.354375, 0.003176, 1.164623],
]


if __name__ == "__main__":
    result = move_arm_joints(CANDIDATE_ANGLES[0])
    print(result)
