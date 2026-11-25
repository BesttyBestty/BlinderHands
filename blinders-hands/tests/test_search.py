from blinders_hands.arm import MockArm
from blinders_hands.camera import ImageSequenceCamera
from blinders_hands.demo import make_demo_frames
from blinders_hands.poses import PoseLibrary, RefineConfig, View
from blinders_hands.search import ActiveSearcher, refine_joints
from blinders_hands.vision.recognizer import ScriptedRecognizer
from blinders_hands.vision.text_detector import OpenCVTextDetector, ViewAssessment

BASE = [-0.37, 1.11, 0.08, -1.01, -1.35, 0.0, 1.16]
VIEWS = [View("正面", BASE), View("侧面", BASE[:6] + [0.4])]
GOOD = {"name": "布洛芬", "production_date": "2025年03月", "expiry": "2027年02月"}


def make(frames, responses, **kw):
    arm = MockArm()
    s = ActiveSearcher(arm, ImageSequenceCamera(frames), OpenCVTextDetector(), ScriptedRecognizer(responses),
                       PoseLibrary(VIEWS, RefineConfig()), **kw)
    return s, arm


def test_skips_blank_side_refines_tilted_then_reads():
    blank, tilted, readable = make_demo_frames()
    s, arm = make([blank, tilted, readable], [GOOD])
    r = s.run("check_expiry")
    assert r.success and r.dates.expiry.year == 2027
    assert r.cloud_calls == 1  # 空白面和歪的那一面都没有浪费云端调用
    moves = [h for h in arm.history if h[0] == "move"]
    assert len(moves) == 3  # 正面 → 侧面 → 微调一次
    assert moves[2][1][6] != moves[1][1][6]  # 微调动的是腕部自转


def test_wrong_info_moves_to_next_view():
    _, _, readable = make_demo_frames()
    s, _ = make([readable], [{"name": "布洛芬"}, GOOD])
    r = s.run("check_expiry")
    assert r.success and r.cloud_calls == 2 and r.steps[-1].view == "侧面"


def test_cloud_budget():
    _, _, readable = make_demo_frames()
    s, _ = make([readable], [{}, {}], max_cloud_calls=1)
    r = s.run("check_expiry")
    assert not r.success and r.cloud_calls == 1


def test_empty_library():
    s = ActiveSearcher(MockArm(), ImageSequenceCamera(make_demo_frames()), OpenCVTextDetector(),
                       ScriptedRecognizer(), PoseLibrary([]))
    assert "姿态库是空的" in s.run("identify").reason


def test_refine_is_bounded():
    a = ViewAssessment(True, False, skew_deg=80, center_offset=(1.0, -1.0))
    new = refine_joints(BASE, a, RefineConfig(max_step=0.2))
    assert all(abs(n - b) <= 0.2 + 1e-9 for n, b in zip(new, BASE))
    assert refine_joints(BASE, ViewAssessment(True, False), RefineConfig()) is None
