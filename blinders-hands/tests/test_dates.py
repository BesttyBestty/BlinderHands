from datetime import date

from blinders_hands.dates import judge_expiry, parse_dates

TODAY = date(2026, 11, 26)


def test_expiry_year_month_chinese():
    info = parse_dates("生产日期：2025年03月 有效期至：2027年02月")
    assert info.production == date(2025, 3, 1)
    assert info.expiry == date(2027, 2, 28)  # 只有年月时按当月最后一天


def test_expiry_full_date_dots():
    assert parse_dates("有效期至 2026.11.20").expiry == date(2026, 11, 20)


def test_expiry_compact():
    assert parse_dates("有效期至20270315").expiry == date(2027, 3, 15)


def test_exp_english_month_year():
    assert parse_dates("EXP: 03/2027").expiry == date(2027, 3, 31)


def test_shelf_life_estimate():
    info = parse_dates("生产日期 2025-03-15 保质期 24个月")
    assert info.expiry == date(2027, 3, 14)
    assert info.expiry_is_estimated


def test_shelf_life_years():
    assert parse_dates("生产日期:2024-06-01 有效期 2年").expiry == date(2026, 5, 31)


def test_production_date_alone_is_not_expiry():
    assert not parse_dates("生产日期 2025-03-15").found


def test_judge():
    assert judge_expiry(parse_dates("有效期至 2026-10-31"), TODAY).status == "expired"
    assert judge_expiry(parse_dates("有效期至 2026-12-10"), TODAY).status == "expiring_soon"
    assert judge_expiry(parse_dates("有效期至 2027-12"), TODAY).status == "valid"
    assert judge_expiry(parse_dates("什么都没有"), TODAY).status == "unknown"
