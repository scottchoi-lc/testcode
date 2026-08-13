"""API smoke tests. The heavy pipeline (`run_pipeline`, which lazily loads
torch/transformers and downloads model weights on first use) and the
detection model are monkeypatched out so these run fast, offline, and
without any ML dependencies installed."""
import io
import json

import app.main as main_module
from app.schemas import ActionLabel, ActionSegment, AnalysisResult, JobStatus
from fastapi.testclient import TestClient


class _FakeDetection:
    def __init__(self, box, score=0.9):
        self.box = box
        self.score = score


def _upload_video(client: TestClient, monkeypatch, duration=8.0, width=1280, height=720):
    """Upload a fake video via /videos, with probe_video mocked so this
    doesn't need a real decodable video file. Returns the video_id."""
    monkeypatch.setattr(main_module, "probe_video", lambda path: (duration, width, height))
    resp = client.post(
        "/videos",
        files={"video": ("clip.mp4", io.BytesIO(b"\x00" * 1024), "video/mp4")},
    )
    assert resp.status_code == 200
    return resp.json()["video_id"]


def test_health():
    client = TestClient(main_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unknown_job_returns_404():
    client = TestClient(main_module.app)
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_upload_video_rejects_non_video_upload():
    client = TestClient(main_module.app)
    resp = client.post(
        "/videos",
        files={"video": ("clip.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_video_returns_id_and_probed_metadata(monkeypatch):
    client = TestClient(main_module.app)
    monkeypatch.setattr(main_module, "probe_video", lambda path: (12.5, 1920, 1080))
    resp = client.post(
        "/videos",
        files={"video": ("clip.mp4", io.BytesIO(b"\x00" * 1024), "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["duration_seconds"] == 12.5
    assert body["frame_width"] == 1920
    assert body["frame_height"] == 1080
    assert body["video_id"]


def test_preview_frame_unknown_video_id_returns_404():
    client = TestClient(main_module.app)
    resp = client.post("/videos/does-not-exist/preview-frame", data={"timestamp": "1.0"})
    assert resp.status_code == 404


def test_preview_frame_returns_detected_people(monkeypatch):
    import numpy as np

    client = TestClient(main_module.app)
    video_id = _upload_video(client, monkeypatch)

    fake_image = np.zeros((100, 200, 3), dtype="uint8")
    monkeypatch.setattr(main_module, "extract_frame_at", lambda path, ts: fake_image)

    class _FakeDetectionModel:
        def detect_ball_and_players(self, image):
            return [], [_FakeDetection((10.0, 20.0, 50.0, 90.0), score=0.87)]

    monkeypatch.setattr(main_module, "get_detection_model", lambda: _FakeDetectionModel())

    resp = client.post(f"/videos/{video_id}/preview-frame", data={"timestamp": "2.5"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == video_id
    assert body["timestamp"] == 2.5
    assert body["frame_width"] == 200
    assert body["frame_height"] == 100
    assert len(body["people"]) == 1
    assert body["people"][0] == {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 90.0, "score": 0.87}
    assert body["image_base64"]  # non-empty JPEG payload


def test_analyze_unknown_video_id_returns_404():
    client = TestClient(main_module.app)
    resp = client.post("/analyze", data={"video_id": "does-not-exist"})
    assert resp.status_code == 404


def test_analyze_end_to_end_with_stubbed_pipeline(monkeypatch):
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

    def fake_run_pipeline(video_path, progress_cb=None, selected_player_box=None, selected_timestamp=None):
        if progress_cb:
            progress_cb(1.0)
        return fake_result

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)

    client = TestClient(main_module.app)
    video_id = _upload_video(client, monkeypatch)

    resp = client.post("/analyze", data={"video_id": video_id})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == JobStatus.QUEUED.value

    # TestClient runs background tasks synchronously before returning the response.
    status_resp = client.get(f"/jobs/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == JobStatus.DONE.value
    assert body["result"]["segments"][0]["label"] == "dribbling"


def test_analyze_passes_selected_box_and_timestamp_to_pipeline(monkeypatch):
    fake_result = AnalysisResult(duration_seconds=1.0, fps_analyzed=6.0, segments=[], summary={})
    captured = {}

    def fake_run_pipeline(video_path, progress_cb=None, selected_player_box=None, selected_timestamp=None):
        captured["selected_player_box"] = selected_player_box
        captured["selected_timestamp"] = selected_timestamp
        return fake_result

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)

    client = TestClient(main_module.app)
    video_id = _upload_video(client, monkeypatch)

    resp = client.post(
        "/analyze",
        data={
            "video_id": video_id,
            "selected_box": json.dumps([10.0, 20.0, 50.0, 90.0]),
            "selected_timestamp": "3.5",
        },
    )
    assert resp.status_code == 200
    client.get(f"/jobs/{resp.json()['job_id']}")
    assert captured["selected_player_box"] == (10.0, 20.0, 50.0, 90.0)
    assert captured["selected_timestamp"] == 3.5


def test_analyze_ignores_malformed_selected_box(monkeypatch):
    fake_result = AnalysisResult(duration_seconds=1.0, fps_analyzed=6.0, segments=[], summary={})
    captured = {}

    def fake_run_pipeline(video_path, progress_cb=None, selected_player_box=None, selected_timestamp=None):
        captured["selected_player_box"] = selected_player_box
        captured["selected_timestamp"] = selected_timestamp
        return fake_result

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)

    client = TestClient(main_module.app)
    video_id = _upload_video(client, monkeypatch)

    resp = client.post(
        "/analyze",
        data={"video_id": video_id, "selected_box": "not json", "selected_timestamp": "3.5"},
    )
    assert resp.status_code == 200
    client.get(f"/jobs/{resp.json()['job_id']}")
    assert captured["selected_player_box"] is None
    assert captured["selected_timestamp"] is None


def test_parse_selected_box_valid():
    assert main_module._parse_selected_box(json.dumps([1, 2, 3, 4])) == (1.0, 2.0, 3.0, 4.0)


def test_parse_selected_box_none_input():
    assert main_module._parse_selected_box(None) is None


def test_parse_selected_box_wrong_length():
    assert main_module._parse_selected_box(json.dumps([1, 2, 3])) is None


def test_parse_selected_box_malformed_json():
    assert main_module._parse_selected_box("{not valid") is None
