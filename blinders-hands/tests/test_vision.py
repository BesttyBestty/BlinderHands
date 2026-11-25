from blinders_hands.demo import make_demo_frames
from blinders_hands.vision.recognizer import extract_json
from blinders_hands.vision.text_detector import OpenCVTextDetector, assess_view


def test_demo_frames_assessment():
    blank, tilted, readable = make_demo_frames()
    d = OpenCVTextDetector()
    assert not assess_view(blank, d).has_text
    a = assess_view(tilted, d)
    assert a.has_text and not a.readable and abs(a.skew_deg) > 20
    assert assess_view(readable, d).readable


def test_extract_json_from_fenced_output():
    assert extract_json('好的：\n```json\n{"name": "布洛芬"}\n```') == {"name": "布洛芬"}
    assert extract_json("没有 JSON") == {}
