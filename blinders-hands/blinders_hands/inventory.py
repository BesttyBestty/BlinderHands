"""药箱记录：每次读到一盒药的有效期就记下来，之后可以直接回答“哪些药快过期了”，并主动提醒。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class Item:
    name: str
    expiry: str          # ISO 日期
    estimated: bool = False
    location: str = ""   # “在您左边、靠近您”
    last_checked: str = ""


class Inventory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.items: dict[str, Item] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = {k: Item(**v) for k, v in data.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.items.items()}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def record(self, name: str, expiry: date, estimated: bool = False, location: str = "",
               today: date | None = None) -> None:
        if not name:
            return
        with self._lock:
            self.items[name] = Item(name, expiry.isoformat(), estimated, location, (today or date.today()).isoformat())
            self._save()

    def expiring(self, within_days: int = 30, today: date | None = None) -> tuple[list[Item], list[Item]]:
        """返回（已过期, 即将过期），各自按有效期排序。"""
        today = today or date.today()
        expired, soon = [], []
        for it in self.items.values():
            left = (date.fromisoformat(it.expiry) - today).days
            if left < 0:
                expired.append(it)
            elif left <= within_days:
                soon.append(it)
        key = lambda it: it.expiry  # noqa: E731
        return sorted(expired, key=key), sorted(soon, key=key)

    def summary(self, within_days: int = 30, today: date | None = None) -> str:
        if not self.items:
            return "我还没有检查过您的药。说一句“帮我整理一下药箱”，我会把每盒药都看一遍。"
        expired, soon = self.expiring(within_days, today)
        parts = []
        if expired:
            parts.append(f"有{len(expired)}样已经过期了：" + "、".join(it.name for it in expired) + "，建议不要再用")
        if soon:
            today = today or date.today()
            desc = [f"{it.name}还剩{(date.fromisoformat(it.expiry) - today).days}天" for it in soon]
            parts.append(f"有{len(soon)}样快过期了：" + "、".join(desc))
        if not parts:
            return f"我记录的{len(self.items)}样药都在有效期内，最近一个月没有要过期的。"
        return "。".join(parts) + "。"
