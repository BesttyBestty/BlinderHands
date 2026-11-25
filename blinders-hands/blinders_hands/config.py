"""运行配置：全部从环境变量读取，不在代码里写死任何密钥。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # 机械臂（松灵 NERO）：can = 官方 SDK pyAgxArm（支持视觉抓取、夹爪、急停）；http = Web 后台接口（示教轨迹）
    arm_backend: str = field(default_factory=lambda: os.environ.get("ARM_BACKEND", "http"))
    can_channel: str = field(default_factory=lambda: os.environ.get("CAN_CHANNEL", "can0"))
    can_interface: str = field(default_factory=lambda: os.environ.get("CAN_INTERFACE", "socketcan"))
    nero_firmware: str = field(default_factory=lambda: os.environ.get("NERO_FIRMWARE", "default"))
    speed_percent: int = field(default_factory=lambda: int(os.environ.get("ARM_SPEED_PERCENT", "30")))
    crash_rating: int = field(default_factory=lambda: int(os.environ.get("ARM_CRASH_RATING", "3")))
    arm_base_url: str = field(default_factory=lambda: os.environ.get("ARM_BASE_URL", "http://10.90.0.150"))
    arm_token: str = field(default_factory=lambda: os.environ.get("ARM_TOKEN", ""))
    arm_move_wait: float = field(default_factory=lambda: float(os.environ.get("ARM_MOVE_WAIT", "2.0")))
    arm_grasp_wait: float = field(default_factory=lambda: float(os.environ.get("ARM_GRASP_WAIT", "3.0")))

    # 摄像头：手眼相机（看机械臂手里的物体）+ 可选的全局相机（看整个桌面）
    hand_camera_index: int = field(default_factory=lambda: int(os.environ.get("HAND_CAMERA_INDEX", os.environ.get("CAMERA_INDEX", "0"))))
    scene_camera_index: int = field(default_factory=lambda: int(os.environ.get("SCENE_CAMERA_INDEX", "-1")))

    # 深度相机（视觉抓取用，同时充当全局相机）：none / realsense
    depth_camera: str = field(default_factory=lambda: os.environ.get("DEPTH_CAMERA", "none"))

    # 云端多模态模型（阿里云 DashScope，OpenAI 兼容接口）
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))
    dashscope_base_url: str = field(default_factory=lambda: os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    vl_model: str = field(default_factory=lambda: os.environ.get("VL_MODEL", "qwen-vl-plus"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "qwen-turbo"))

    # 端侧文字检测：opencv（任何电脑可跑）或 rdk（地瓜 RDK BPU）
    detector_backend: str = field(default_factory=lambda: os.environ.get("DETECTOR_BACKEND", "opencv"))
    rdk_det_model: str = field(default_factory=lambda: os.environ.get("RDK_DET_MODEL", ""))

    # 主动搜索参数
    max_refine_steps: int = field(default_factory=lambda: int(os.environ.get("MAX_REFINE_STEPS", "2")))
    max_cloud_calls: int = field(default_factory=lambda: int(os.environ.get("MAX_CLOUD_CALLS", "4")))
    min_sharpness: float = field(default_factory=lambda: float(os.environ.get("MIN_SHARPNESS", "80")))

    # 数据文件
    poses_file: Path = field(default_factory=lambda: Path(os.environ.get("POSES_FILE", ROOT / "config" / "poses.json")))
    objects_file: Path = field(default_factory=lambda: Path(os.environ.get("OBJECTS_FILE", ROOT / "config" / "objects.json")))
    grasp_file: Path = field(default_factory=lambda: Path(os.environ.get("GRASP_FILE", ROOT / "config" / "grasp.json")))
    handeye_file: Path = field(default_factory=lambda: Path(os.environ.get("HANDEYE_FILE", ROOT / "config" / "handeye.json")))
    inventory_file: Path = field(default_factory=lambda: Path(os.environ.get("INVENTORY_FILE", ROOT / "data" / "inventory.json")))
    max_views: int = field(default_factory=lambda: int(os.environ.get("MAX_VIEWS", "8")))
    handover_timeout: float = field(default_factory=lambda: float(os.environ.get("HANDOVER_TIMEOUT", "30")))
    capture_dir: Path = field(default_factory=lambda: Path(os.environ.get("CAPTURE_DIR", ROOT / "captures")))

    # 模拟模式：没有机械臂、摄像头时也能把整条链路跑通
    mock: bool = field(default_factory=lambda: _env_bool("BH_MOCK", False))
    # 服务端是否也用 macOS `say` 播报（默认关，手机网页负责播报，避免说两遍）
    speak_on_server: bool = field(default_factory=lambda: _env_bool("SPEAK_ON_SERVER", False))
