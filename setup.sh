#!/bin/bash
set -e

echo "=== COMP2 Setup ==="

# Create directories
mkdir -p uploads results gallery checkpoints

# Python venv
if [ ! -d "venv" ]; then
  echo "[1/5] Creating virtual environment..."
  python3 -m venv venv
else
  echo "[1/5] venv already exists"
fi

source venv/bin/activate

# Install dependencies
echo "[2/5] Installing Python packages..."
pip install --upgrade pip -q
pip install torch torchvision -q

# Install SAM2 from git (avoids PyPI issues)
echo "[3/5] Installing SAM2 from git..."
pip install git+https://github.com/facebookresearch/sam2.git -q

# Install remaining deps
pip install -r requirements.txt -q

# Download YOLO model
if [ ! -f "yolo11m-seg.pt" ]; then
  echo "[4/5] Downloading YOLO11m-seg model (~44MB)..."
  python3 -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
else
  echo "[4/5] YOLO model already exists"
fi

# Download SAM2 checkpoint
if [ ! -f "checkpoints/sam2.1_hiera_small.pt" ]; then
  echo "[5/5] Downloading SAM2 checkpoint (~176MB)..."
  python3 -c "
import urllib.request, os
url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'
os.makedirs('checkpoints', exist_ok=True)
urllib.request.urlretrieve(url, 'checkpoints/sam2.1_hiera_small.pt')
print('Done')
"
else
  echo "[5/5] SAM2 checkpoint already exists"
fi

echo ""
echo "=== Setup complete ==="
echo "Run:  source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "Open: http://localhost:8000"
