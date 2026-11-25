# 部署到地瓜 RDK

盲手在 RDK 上的分工：**主动视觉闭环里每一帧都要做的事（文字检测、视角评估、微调决策）放在 RDK 端侧**；只有确认“这一面可以读”之后，才把一张图送到云端识别。机械臂每动一下就能立刻判断，不用等网络往返。

## 1. 环境

在 RDK X5 上（系统镜像自带 Python 3 和 `hobot_dnn`）：

```bash
git clone https://github.com/SSShichuanyi/-Blinders-hands.git
cd -Blinders-hands && git checkout rdk-hackathon
pip3 install -r requirements.txt
```

NERO 通过 USB-CAN 模块接到 RDK X5（`ARM_BACKEND=can`，Linux 下用 `socketcan`，先按 pyAgxArm 文档把 CAN 口配置为 `can0`、1 Mbps）；深度相机和手眼相机接 RDK 的 USB 3.0 口。RDK 通过 Wi-Fi 联网，给手机提供网页。

```bash
pip3 install -r requirements-hw.txt
sudo ip link set can0 up type can bitrate 1000000
```

## 2. 文字检测模型

`RDKTextDetector` 需要一个 DBNet 类文字检测模型（输入 NV12，输出 1×1×H×W 的文字概率图），例如 PaddleOCR 的检测模型：

1. 把检测模型导出为 ONNX；
2. 用地瓜的模型转换工具链，按 RDK X5 的 BPU 量化编译成 `.bin`（输入类型选 NV12）；
3. 把 `.bin` 拷到板子上，然后：

```bash
export DETECTOR_BACKEND=rdk
export RDK_DET_MODEL=/path/to/text_det.bin
python3 -m blinders_hands.server
```

没有转换好模型之前，`DETECTOR_BACKEND=opencv`（默认）在 RDK 的 CPU 上同样可以跑。

## 3. 相机

USB 相机直接用 `HAND_CAMERA_INDEX` / `SCENE_CAMERA_INDEX` 指定，先跑 `python3 tools/check_cameras.py` 确认索引。

## 说明

- `RDKTextDetector` 使用 RDK X5 的 `hobot_dnn.pyeasy_dnn` 接口，模型输入尺寸从模型文件中自动读取；换用其他检测模型时，按其输出布局调整后处理。
- RDK S100 使用不同的推理运行时，部署到 S100 时需替换 `RDKTextDetector` 中的模型加载与推理调用。
