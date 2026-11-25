"""引导使用者的手：全局相机同时找到“手”和“目标物”，一句一句告诉他往哪边挪，直到摸到。

画面约定与 direction_phrase 一致：相机从使用者身后俯视，画面左边是使用者的左边，画面下方离使用者近。
"""

from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass
class GuideStep:
    say: str
    done: bool
    distance: float | None = None


def _center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def guide_instruction(hand: BBox | None, target: BBox | None, reach: float = 0.06) -> GuideStep:
    if target is None:
        return GuideStep("我现在看不到它了，请把手先收回来一点。", False)
    if hand is None:
        return GuideStep("请把手伸到桌面上方，我来告诉您往哪边挪。", False)

    hx, hy = _center(hand)
    tx, ty = _center(target)
    dx, dy = tx - hx, ty - hy  # dx>0：目标在手的右边；dy<0：目标比手更靠里（远离使用者）
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if _overlap(hand, target) or dist < reach:
        return GuideStep("到了，就在您手下面，可以拿了。", True, dist)

    def amount(d: float) -> str:
        return "一点点" if d < 0.08 else "一点" if d < 0.2 else ""

    parts = []
    # 先纠正偏得多的那个方向，一次最多说两个方向
    axes = sorted([("x", dx), ("y", dy)], key=lambda t: -abs(t[1]))
    for axis, d in axes:
        if abs(d) < reach * 0.6:
            continue
        if axis == "x":
            parts.append(("往右" if d > 0 else "往左") + amount(abs(d)))
        else:
            parts.append(("往前" if d < 0 else "往回收") + amount(abs(d)))
    return GuideStep("，".join(parts[:2]) + "。", False, dist)
