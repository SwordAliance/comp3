"""
Tests for COMP2 pipeline and API.
Run: python -m pytest tests/ -v
"""

import numpy as np
import cv2
import pytest


# ---- Test imports ----

def test_import_fastapi_app():
    """FastAPI app module imports without error."""
    from app.main import app
    assert app is not None
    assert app.title == "Car Defect Detection"


def test_import_pipeline():
    """Pipeline module imports without error."""
    from app.pipeline import process_image, detect_car, segment_car, detect_defects_simple
    assert callable(process_image)
    assert callable(detect_car)
    assert callable(segment_car)
    assert callable(detect_defects_simple)


def test_import_preprocessing():
    """Preprocessing module imports without error."""
    from app.preprocessing import remove_glare, crop_by_mask, classify_defects
    assert callable(remove_glare)
    assert callable(crop_by_mask)
    assert callable(classify_defects)


def test_import_models():
    """Models module imports and has expected functions."""
    from app.models import get_device, clear_vram, unload_model, get_yolo, get_sam2
    assert callable(get_device)
    assert callable(clear_vram)
    assert callable(unload_model)
    assert get_device() in ("cuda", "cpu")


# ---- Test API endpoints via TestClient ----

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_index_page(client):
    """GET / returns the index page."""
    resp = client.get("/")
    assert resp.status_code == 200


def test_panel_page(client):
    """GET /panel.html returns the panel page."""
    resp = client.get("/panel.html")
    assert resp.status_code == 200


def test_list_photos(client):
    """GET /api/photos returns a list."""
    resp = client.get("/api/photos")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_upload_photo(client):
    """POST /api/photos with base64 data creates a photo."""
    # Minimal 1x1 white JPEG
    img = np.ones((1, 1, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", img)
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode()

    resp = client.post("/api/photos", json={"data": f"data:image/jpeg;base64,{b64}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    photo_id = data["id"]

    # Cleanup
    client.delete(f"/api/photos/{photo_id}")


def test_delete_photo_not_found(client):
    """DELETE /api/photos/nonexistent returns 404."""
    resp = client.delete("/api/photos/nonexistent")
    assert resp.status_code == 404


def test_upload_image_rejects_non_image(client):
    """POST /api/upload rejects non-image files."""
    resp = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_result_not_found(client):
    """GET /api/result/nonexistent returns 404."""
    resp = client.get("/api/result/nonexistent")
    assert resp.status_code == 404


# ---- Test classify_defects with synthetic data ----

def test_classify_defects_empty():
    """Empty anomaly map returns no defects."""
    from app.preprocessing import classify_defects
    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    crop = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = classify_defects(anomaly_map, crop)
    assert result == []


def test_classify_defects_spot():
    """A circular dark defect is classified as a spot."""
    from app.preprocessing import classify_defects
    anomaly_map = np.zeros((200, 200), dtype=np.float32)
    # Draw a circular defect
    cv2.circle(anomaly_map, (100, 100), 15, 0.8, -1)

    # Dark crop region matching the defect
    crop = np.ones((200, 200, 3), dtype=np.uint8) * 180
    cv2.circle(crop, (100, 100), 15, (40, 40, 40), -1)

    result = classify_defects(anomaly_map, crop, threshold=0.01)
    assert len(result) >= 1
    assert result[0]["type"] == "spot"
    assert "bbox" in result[0]


def test_classify_defects_scratch():
    """An elongated defect is classified as a scratch."""
    from app.preprocessing import classify_defects
    anomaly_map = np.zeros((200, 200), dtype=np.float32)
    # Draw an elongated scratch (aspect > 3)
    cv2.rectangle(anomaly_map, (50, 95), (180, 105), 0.8, -1)

    crop = np.ones((200, 200, 3), dtype=np.uint8) * 180

    result = classify_defects(anomaly_map, crop, threshold=0.01)
    assert len(result) >= 1
    assert result[0]["type"] == "scratch"


# ---- Test image resize logic ----

def test_resize_large_image():
    """Images larger than 2048px are resized proportionally."""
    # Simulate the resize logic from process_image
    MAX_SIDE = 2048

    # 4000x3000 image
    h, w = 3000, 4000
    image = np.zeros((h, w, 3), dtype=np.uint8)

    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    assert image.shape[1] <= MAX_SIDE
    assert image.shape[0] <= MAX_SIDE
    # Check proportions are preserved
    assert abs(image.shape[1] / image.shape[0] - 4000 / 3000) < 0.01


def test_no_resize_small_image():
    """Images within 2048px are not resized."""
    MAX_SIDE = 2048
    h, w = 1080, 1920
    image = np.zeros((h, w, 3), dtype=np.uint8)

    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    assert image.shape[0] == 1080
    assert image.shape[1] == 1920
