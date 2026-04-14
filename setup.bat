@echo off
setlocal

echo === COMP2 Setup (Windows) ===

:: Create directories
if not exist uploads mkdir uploads
if not exist results mkdir results
if not exist gallery mkdir gallery
if not exist checkpoints mkdir checkpoints

:: Python venv
if not exist venv (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/5] venv already exists
)

call venv\Scripts\activate.bat

:: Install PyTorch with CUDA 12.4
echo [2/5] Installing PyTorch with CUDA 12.4...
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

:: Install SAM2 from git (no PyPI wheel for Windows)
echo [3/5] Installing SAM2 from git...
pip install git+https://github.com/facebookresearch/sam2.git -q

:: Install remaining dependencies
echo [3/5] Installing other dependencies...
pip install -r requirements.txt -q

:: Download YOLO model
if not exist yolo11m-seg.pt (
    echo [4/5] Downloading YOLO11m-seg model (~44MB)...
    python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
) else (
    echo [4/5] YOLO model already exists
)

:: Download SAM2 checkpoint
if not exist checkpoints\sam2.1_hiera_small.pt (
    echo [5/5] Downloading SAM2 checkpoint (~176MB)...
    python -c "import urllib.request, os; os.makedirs('checkpoints', exist_ok=True); urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt', 'checkpoints/sam2.1_hiera_small.pt'); print('Done')"
) else (
    echo [5/5] SAM2 checkpoint already exists
)

echo.
echo === Setup complete ===
echo Run:  venv\Scripts\activate.bat ^&^& uvicorn app.main:app --host 0.0.0.0 --port 8000
echo Open: http://localhost:8000

endlocal
