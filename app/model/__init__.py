"""
Lazy model loading with singleton pattern and VRAM management.
Models are loaded on demand and unloaded after use to save VRAM.
"""

import gc
import torch
import logging

logger = logging.getLogger(__name__)

_models = {}
_device = "cpu"


def get_device():
    return _device


def clear_vram():
    """Free memory between pipeline stages."""
    gc.collect()


def unload_model(name: str):
    """Unload a specific model from memory."""
    if name in _models:
        del _models[name]
        clear_vram()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(f"Unloaded {name}, VRAM free: {free/1e9:.1f}GB")
        else:
            logger.info(f"Unloaded {name} (CPU mode)")


def unload_all():
    """Unload all models."""
    _models.clear()
    clear_vram()


def get_yolo():
    """Load YOLO11m-seg for car detection + coarse segmentation."""
    if "yolo" not in _models:
        from ultralytics import YOLO
        logger.info("Loading YOLO11m-seg...")
        model = YOLO("yolo11m-seg.pt")
        model.to(_device)
        _models["yolo"] = model
        logger.info("YOLO11m-seg loaded")
    return _models["yolo"]


def get_sam2():
    """Load SAM2 hiera-small for precise segmentation."""
    if "sam2" not in _models:
        logger.info("Loading SAM2 hiera-small...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = _ensure_sam2_checkpoint()
        model_cfg = "configs/sam2.1/sam2.1_hiera_s"

        sam2_model = build_sam2(
            model_cfg,
            checkpoint,
            device=_device,
        )
        predictor = SAM2ImagePredictor(sam2_model)
        _models["sam2"] = predictor
        logger.info("SAM2 loaded")
    return _models["sam2"]


def _ensure_sam2_checkpoint():
    """Download SAM2 checkpoint if not present."""
    import os
    from pathlib import Path

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "sam2.1_hiera_small.pt"

    if not ckpt_path.exists():
        logger.info("Downloading SAM2 checkpoint...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        torch.hub.download_url_to_file(url, str(ckpt_path))
        logger.info("SAM2 checkpoint downloaded")

    return str(ckpt_path)
