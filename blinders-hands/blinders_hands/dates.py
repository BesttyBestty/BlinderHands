"""从包装文字里解析生产日期 / 有效期，并判断是否过期。

覆盖国内药品、食品包装上常见的写法：
- 有效期至 2027年03月 / 有效期至：2027.03.15 / 有效期至20270315
- EXP 2027-03 / EXP: 03/2027
- 生产日期 2025-03-15 + 保质期 24 个月 / 有效期 2 年（推算）
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

_YMD = r"(?P<y>20\d{2})\s*[-./年]?\s*(?P<m>1[0-2]|0?[1-9])(?!\d)\s*(?:[-./月]\s*(?P<d>3[01]|[12]\d|0?[1-9])(?!\d)\s*日?)?"
_YMD_COMPACT = r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])?"
_MY = r"(?P<m>1[0-2]|0?[1-9])\s*[/.]\s*(?P<y>20\d{2})"

_EXPIRY_KEYS = r"(?:有效期至|有效期到|有效日期|失效日期|有效期|保质期至|保质期到|EXP(?:IRY)?(?:\s*DATE)?|USE\s*BY|BEST\s*BEFORE)"
_PRODUCTION_KEYS = r"(?:生产日期|生产批号日期|制造日期|MFG(?:\s*DATE)?|MFD|PROD(?:UCTION)?\s*DATE)"
_SEP = r"\s*[:：]?\s*"


@dataclass
class DateInfo:
    production: date | None = None
    expiry: date | None = None
    expiry_is_estimated: bool = False  # 由生产日期 + 保质期推算得出

    @property
    def found(self) -> bool:
        return self.expiry is not None


def _end_of_month(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _to_date(y: str, m: str, d: str | None, month_end: bool) -> date | None:
    try:
        yi, mi = int(y), int(m)
        if d:
            return date(yi, mi, int(d))
        return _end_of_month(yi, mi) if month_end else date(yi, mi, 1)
    except ValueError:
        return None


def _find_date_after(keys: str, text: str, month_end: bool) -> date | None:
    for pattern in (_YMD_COMPACT + r"(?!\d)", _YMD, _MY):
        m = re.search(keys + _SEP + pattern, text, flags=re.IGNORECASE)
        if m:
            got = _to_date(m.group("y"), m.group("m"), m.group("d") if "d" in m.groupdict() else None, month_end)
            if got:
                return got
    return None


def _shelf_life_months(text: str) -> int | None:
    m = re.search(r"(?:保质期|有效期)\s*[:：]?\s*(\d{1,3})\s*(个月|月|年)", text)
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if m.group(2) == "年" else n


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    y, mo = d.year + total // 12, total % 12 + 1
    return date(y, mo, min(d.day, calendar.monthrange(y, mo)[1]))


def parse_dates(text: str) -> DateInfo:
    info = DateInfo()
    if not text:
        return info
    normalized = text.replace("　", " ").replace("：", ":")
    info.production = _find_date_after(_PRODUCTION_KEYS, normalized, month_end=False)
    info.expiry = _find_date_after(_EXPIRY_KEYS, normalized, month_end=True)
    if info.expiry is None and info.production is not None:
        months = _shelf_life_months(normalized)
        if months:
            # 惯例：有效期 N 个月，到期日为生产日期加 N 个月的前一天
            info.expiry = _add_months(info.production, months) - timedelta(days=1)
            info.expiry_is_estimated = True
    return info


@dataclass
class ExpiryVerdict:
    status: str  # "valid" | "expiring_soon" | "expired" | "unknown"
    message: str
    expiry: date | None = None


def judge_expiry(info: DateInfo, today: date | None = None, soon_days: int = 30) -> ExpiryVerdict:
    today = today or date.today()
    if not info.found:
        return ExpiryVerdict("unknown", "没有找到有效期。")
    exp = info.expiry
    when = f"{exp.year}年{exp.month}月{exp.day}日"
    note = "（根据生产日期和保质期推算）" if info.expiry_is_estimated else ""
    left = (exp - today).days
    if left < 0:
        return ExpiryVerdict("expired", f"已经过期了，有效期到{when}{note}，不建议使用。", exp)
    if left <= soon_days:
        return ExpiryVerdict("expiring_soon", f"还能用，但快过期了，有效期到{when}{note}，还剩{left}天。", exp)
    return ExpiryVerdict("valid", f"在有效期内，可以使用，有效期到{when}{note}。", exp)
