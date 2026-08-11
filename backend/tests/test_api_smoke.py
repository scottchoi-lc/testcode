"""API smoke tests. The heavy pipeline (`run_pipeline`, which lazily loads
torch/transformers and downloads model weights on first use) is monkeypatched
out so these run fast, offline, and without any ML dependencies installed."""
import io

import app.main as main_module
from app.schemas import ActionLabel, ActionSegment, AnalysisResult, JobStatus
from fastapi.testclient import TestClient


def test_health():
    client = TestClient(main_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unknown_job_returns_404():
    client = TestClient(main_module.app)
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_analyze_rejects_non_video_upload():
    client = TestClient(main_module.app)
    resp = client.post(
        "/analyze",
        files={"video": ("clip.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_end_to_end_with_stubbed_pipeline(monkeypatch, tmp_path):
    fake_result = AnalysisResult(
        duration_seconds=2.0,
        fps_analyzed=6.0,
        segments=[
            ActionSegment(
                start_time=0.0, end_time=2.0, label=ActionLabel.DRIBBLING, confidence=0.9
            )
        ],
        summary={label.value: 0.0 for label in ActionLabel} | {"dribbling": 2.0},
    )

    def fake_run_pipeline(video_path, progress_cb=None):
        if progress_cb:
            progress_cb(1.0)
        return fake_result

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)

    client = TestClient(main_module.app)
    resp = client.post(
        "/analyze",
        files={"video": ("clip.mp4", io.BytesIO(b"\x00" * 1024), "video/mp4")},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == JobStatus.QUEUED.value

    # TestClient runs background tasks synchronously before returning the response.
    status_resp = client.get(f"/jobs/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == JobStatus.DONE.value
    assert body["result"]["segments"][0]["label"] == "dribbling"
