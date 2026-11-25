"""
自动旋转搜索：机械臂依次尝试候选角度，每个角度拍照识别，
直到成功找到有效期文字为止。
这是 demo 的核心自动化闭环，CANDIDATE_ANGLES 需要现场提前标定真实数值。
"""

import time
from arm_control import move_arm_joints, CANDIDATE_ANGLES
from full_pipeline import capture_and_check


def auto_search():
    for i, angles in enumerate(CANDIDATE_ANGLES):
        print(f"尝试角度 {i + 1}/{len(CANDIDATE_ANGLES)}")
        move_arm_joints(angles)
        time.sleep(2)  # 等待机械臂运动到位，可根据实际速度调整

        result = capture_and_check()
        if result and "未能识别" not in result:
            print(f"成功找到日期信息: {result}")
            return result

    print("所有预设角度都未能找到日期信息")
    return None


if __name__ == "__main__":
    auto_search()
