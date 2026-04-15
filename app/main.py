"""
FastAPI server for car defect detection.
"""

import uuid
import json
import base64
import logging
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Defect Detection")

from starlette.middleware.base import BaseHTTPMiddleware

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path.startswith("/gallery"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

app.add_middleware(NoCacheMiddleware)

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
GALLERY_DIR = Path("gallery")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
GALLERY_DIR.mkdir(exist_ok=True)

# Track processing status
_status = {}


def run_pipeline(job_id: str, image_path: str):
    """Run ML pipeline in background."""
    from app.pipeline import process_image

    _status[job_id] = {"status": "processing", "progress": "Запуск пайплайна..."}
    try:
        result_dir = str(RESULT_DIR / job_id)

        def on_progress(msg):
            _status[job_id]["progress"] = msg

        result = process_image(image_path, result_dir, progress_callback=on_progress)

        if "error" in result and result.get("defect_count") is None:
            _status[job_id] = {"status": "error", "error": result["error"]}
        else:
            _status[job_id] = {"status": "done", "result": result}

        # Save result JSON
        with open(Path(result_dir) / "result.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.exception(f"Pipeline failed for {job_id}")
        _status[job_id] = {"status": "error", "error": str(e)}


# ---- Existing endpoints ----

@app.post("/api/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload an image and start processing."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    content = await file.read()
    save_path.write_bytes(content)

    _status[job_id] = {"status": "processing", "progress": "Starting..."}

    # Run pipeline in background
    background_tasks.add_task(run_pipeline, job_id, str(save_path))

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Get processing result."""
    if job_id not in _status:
        raise HTTPException(404, "Job not found")

    return _status[job_id]


@app.get("/api/result/{job_id}/file/{filename}")
async def get_result_file(job_id: str, filename: str):
    """Get a result file (image)."""
    # Sanitize filename
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")

    file_path = RESULT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(file_path)


# ---- Gallery endpoints ----

class PhotoUpload(BaseModel):
    data: str  # base64 data URL


@app.get("/api/photos")
async def list_photos():
    """List all photos in gallery."""
    photos = []
    for f in sorted(GALLERY_DIR.iterdir()):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            photos.append({
                "id": f.stem,
                "d": f"/gallery/{f.name}",
                "t": f.stat().st_mtime,
            })
    # Sort by modification time, newest first
    photos.sort(key=lambda x: x["t"], reverse=True)
    return photos


@app.post("/api/photos")
async def upload_photo(photo: PhotoUpload):
    """Upload a base64 JPEG to gallery."""
    data_url = photo.data
    # Strip data URL header: "data:image/jpeg;base64,..."
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(data_url)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")

    photo_id = str(uuid.uuid4())[:8]
    save_path = GALLERY_DIR / f"{photo_id}.jpg"
    save_path.write_bytes(image_bytes)

    return {"id": photo_id, "status": "ok"}


@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    """Delete a photo from gallery."""
    if ".." in photo_id or "/" in photo_id:
        raise HTTPException(400, "Invalid photo ID")

    # Find photo file (could be .jpg, .jpeg, .png)
    found = False
    for f in GALLERY_DIR.iterdir():
        if f.stem == photo_id:
            f.unlink()
            found = True
            break

    if not found:
        raise HTTPException(404, "Photo not found")

    return {"status": "ok"}


@app.post("/api/photos/{photo_id}/rotate")
async def rotate_photo(photo_id: str):
    """Rotate a gallery photo 90° clockwise."""
    if ".." in photo_id or "/" in photo_id:
        raise HTTPException(400, "Invalid photo ID")

    photo_path = None
    for f in GALLERY_DIR.iterdir():
        if f.stem == photo_id:
            photo_path = f
            break

    if photo_path is None:
        raise HTTPException(404, "Photo not found")

    import cv2
    img = cv2.imread(str(photo_path))
    if img is None:
        raise HTTPException(500, "Failed to read image")

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(photo_path), rotated)

    return {"status": "ok"}


@app.post("/api/analyze/{photo_id}")
async def analyze_photo(photo_id: str, background_tasks: BackgroundTasks):
    """Start ML pipeline on a gallery photo."""
    if ".." in photo_id or "/" in photo_id:
        raise HTTPException(400, "Invalid photo ID")

    # Find photo file
    photo_path = None
    for f in GALLERY_DIR.iterdir():
        if f.stem == photo_id:
            photo_path = f
            break

    if photo_path is None:
        raise HTTPException(404, "Photo not found")

    job_id = str(uuid.uuid4())[:8]
    _status[job_id] = {"status": "processing", "progress": "Запуск..."}

    background_tasks.add_task(run_pipeline, job_id, str(photo_path))

    return {"job_id": job_id, "status": "processing"}


# ---- Page routes ----

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/panel.html")
async def panel():
    return FileResponse("static/panel.html")


# Mount static files and gallery
app.mount("/gallery", StaticFiles(directory="gallery"), name="gallery")
app.mount("/static", StaticFiles(directory="static"), name="static")
