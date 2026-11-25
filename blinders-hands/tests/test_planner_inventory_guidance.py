from datetime import date

from blinders_hands.guidance import guide_instruction
from blinders_hands.inventory import Inventory
from blinders_hands.poses import PoseLibrary, RefineConfig, View
from blinders_hands.search import ViewPlanner
from blinders_hands.vision.text_detector import ViewAssessment

BASE = [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]


def planner():
    return ViewPlanner(PoseLibrary([View("正面", BASE)], RefineConfig(yaw_joint=4, yaw_gain=0.35),
                                   sweep_joint=4, sweep_offsets=[-1.6, -0.8, 0.8, 1.6]))


def test_blank_face_jumps_far():
    p = planner()
    first = p.next()
    assert first.name == "正面"
    p.observe(first, ViewAssessment(False, False))
    nxt = p.next()
    assert abs(nxt.joints[4] - first.joints[4]) > 1.0   # 空白面附近也多半是空白，直接转到远处


def test_text_on_edge_turns_that_way():
    p = planner()
    first = p.next()
    p.observe(first, ViewAssessment(True, False, center_offset=(0.8, 0.0)))  # 字在画面右边
    nxt = p.next()
    assert nxt.joints[4] > first.joints[4]              # yaw_gain>0：往正方向转能把右边的字转过来


def test_inventory(tmp_path):
    inv = Inventory(tmp_path / "inv.json")
    today = date(2026, 11, 26)
    assert "还没有检查过" in inv.summary(today=today)
    inv.record("布洛芬", date(2027, 2, 28), today=today)
    inv.record("阿莫西林", date(2026, 12, 10), today=today)
    inv.record("感冒灵", date(2026, 10, 1), today=today)
    s = inv.summary(today=today)
    assert "1样已经过期了：感冒灵" in s and "阿莫西林还剩14天" in s
    assert len(Inventory(tmp_path / "inv.json").items) == 3   # 重新打开还在


def test_guidance():
    target = (0.45, 0.40, 0.55, 0.50)
    assert guide_instruction(None, target).done is False
    step = guide_instruction((0.05, 0.85, 0.15, 0.95), target)
    assert "往右" in step.say and "往前" in step.say and not step.done
    assert guide_instruction((0.46, 0.42, 0.54, 0.52), target).done
