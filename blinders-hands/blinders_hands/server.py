"""Flask 后端：手机网页语音交互 ↔ 意图理解 ↔ 机械臂任务。

启动：python -m blinders_hands.server
手机访问：https://<电脑或 RDK 的局域网 IP>:5001
（手机浏览器只允许 HTTPS 页面使用麦克风，所以默认用自签名证书启动，首次打开需点“继续访问”）
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .config import Settings
from .intent import NEEDS_TARGET, NO_CONFIRM, TASKS, Intent, IntentParser
from .pipeline import Assistant, build_assistant

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"


@dataclass
class Job:
    id: str
    intent: Intent
    status: str = "running"  # running | waiting_release | done | error
    progress: list[str] = field(default_factory=list)
    message: str = ""
    success: bool = False
    location: str = ""


def create_app(assistant: Assistant | None = None, parser: IntentParser | None = None,
               settings: Settings | None = None) -> Flask:
    settings = settings or Settings()
    assistant = assistant or build_assistant(settings)
    if parser is None:
        key = "" if settings.mock else settings.dashscope_api_key
        parser = IntentParser(key, settings.dashscope_base_url, settings.llm_model)

    app = Flask(__name__, static_folder=None)
    jobs: dict[str, Job] = {}
    arm_lock = threading.Lock()  # 只有一只手，一次只做一件事

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({
            "mock": settings.mock,
            "detector": settings.detector_backend,
            "views": len(assistant.poses.views),
            "scene_camera": assistant.scene_camera is not None,
            "vision_grasp": assistant.can_vision_grasp,
            "state": assistant.state,
            "busy": arm_lock.locked(),
        })

    @app.post("/api/stop")
    def stop():
        assistant.stop()
        return jsonify({"say": "已经停下了。确认安全后，说“继续”。"})

    @app.post("/api/resume")
    def resume():
        assistant.resume()
        return jsonify({"say": "好的，可以继续了。"})

    @app.post("/api/release")
    def release():
        assistant.release()
        return jsonify({"say": "好的，我松手了。"})

    @app.post("/api/guide")
    def guide():
        target = str((request.get_json(silent=True) or {}).get("target", "")).strip()
        if not target:
            return jsonify({"error": "缺少目标物"}), 400
        if arm_lock.locked():
            return jsonify({"say": "我正在忙，请稍等。", "done": False})
        step = assistant.guide_step(target)
        return jsonify({"say": step.say, "done": step.done})

    @app.get("/api/inventory")
    def inventory():
        inv = assistant.inventory
        expired, soon = inv.expiring()
        return jsonify({"summary": inv.summary(), "remind": bool(expired or soon),
                        "items": [vars(it) for it in inv.items.values()]})

    @app.post("/api/understand")
    def understand():
        text = (request.get_json(silent=True) or {}).get("text", "")
        intent = parser.parse(text)
        return jsonify({"understood": intent.understood, "target": intent.target, "task": intent.task,
                        "confirm": intent.confirm, "question": intent.question, "handover": intent.handover,
                        "needs_confirm": intent.task not in NO_CONFIRM})

    @app.post("/api/tasks")
    def start_task():
        body = request.get_json(silent=True) or {}
        target = str(body.get("target", "")).strip()
        task = body.get("task", "identify")
        if task not in TASKS or (task in NEEDS_TARGET and not target):
            return jsonify({"error": "缺少要找的物品或任务类型"}), 400
        if assistant.state == "stopped":
            return jsonify({"error": "已经急停，请先说“继续”"}), 409
        if not arm_lock.acquire(blocking=False):
            return jsonify({"error": "我正在忙上一件事，请稍等"}), 409

        job = Job(uuid.uuid4().hex[:8], Intent(target or "药箱", task, "", str(body.get("question", "")),
                                               bool(body.get("handover", False))))
        jobs[job.id] = job

        def work():
            try:
                def on_state(state: str) -> None:
                    job.status = "waiting_release" if state == "waiting_release" else "running"

                outcome = assistant.run(job.intent, progress=job.progress.append, on_state=on_state)
                job.message, job.success, job.location = outcome.message, outcome.success, outcome.location
                job.status = "done"
            except Exception as exc:  # 任何意外都要变成一句能说出来的话
                log.exception("任务失败")
                job.message, job.status = f"出了点问题：{exc}", "error"
            finally:
                arm_lock.release()

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"id": job.id}), 202

    @app.get("/api/tasks/<job_id>")
    def task_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "没有这个任务"}), 404
        return jsonify({"id": job.id, "status": job.status, "progress": job.progress,
                        "message": job.message, "success": job.success, "location": job.location})

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app()
    port = int(os.environ.get("PORT", "5001"))
    use_https = os.environ.get("BH_HTTPS", "1") not in {"0", "false", "no"}
    ssl = "adhoc" if use_https else None
    app.run(host="0.0.0.0", port=port, ssl_context=ssl, threaded=True)


if __name__ == "__main__":
    main()
