"""
Preprocessing: glare removal, crop by mask, post-filtering.
"""

import cv2
import numpy as np


def detect_glare_mask(image: np.ndarray, v_thresh: int = 220, s_thresh: int = 40) -> np.ndarray:
    """
    Detect glare regions using HSV analysis.
    Glare = high Value + low Saturation.
    Returns binary mask where 255 = glare.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    glare_mask = np.zeros_like(v, dtype=np.uint8)
    glare_mask[(v > v_thresh) & (s < s_thresh)] = 255

    # Dilate slightly to cover glare edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

    return glare_mask


def remove_glare(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove glare via inpainting.
    Returns (cleaned_image, glare_mask).
    """
    glare_mask = detect_glare_mask(image)

    if glare_mask.sum() == 0:
        return image.copy(), glare_mask

    cleaned = cv2.inpaint(image, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return cleaned, glare_mask


def crop_by_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop image to bounding box of mask region.
    Returns (cropped_image_with_alpha, bbox as (x1,y1,x2,y2)).
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image, (0, 0, image.shape[1], image.shape[0])

    x, y, w, h = cv2.boundingRect(coords)

    cropped = image[y:y+h, x:x+w].copy()
    mask_crop = mask[y:y+h, x:x+w]

    # Create BGRA image with transparency outside mask
    if cropped.shape[2] == 3:
        bgra = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    else:
        bgra = cropped.copy()
    bgra[:, :, 3] = mask_crop

    return bgra, (x, y, x + w, y + h)


def suppress_glare_anomalies(anomaly_map: np.ndarray, glare_mask: np.ndarray,
                              suppression_factor: float = 0.1) -> np.ndarray:
    """
    Suppress false positives in glare regions.
    Multiply anomaly scores by suppression_factor where glare was detected.
    """
    result = anomaly_map.copy()
    if glare_mask.shape[:2] != anomaly_map.shape[:2]:
        glare_mask = cv2.resize(glare_mask, (anomaly_map.shape[1], anomaly_map.shape[0]))

    glare_regions = glare_mask > 127
    result[glare_regions] *= suppression_factor

    return result


def classify_defects(anomaly_map: np.ndarray, original_crop: np.ndarray,
                     threshold: float = 0.15) -> list[dict]:
    """
    Classify detected paint defects.
    Types: 'spot' (dark spot/foreign particle), 'scratch' (elongated scratch).
    Only considers anomaly regions that already passed the threshold in the detector.
    """
    if anomaly_map.max() == 0:
        return []

    if anomaly_map.max() > 1.0:
        norm_map = anomaly_map / anomaly_map.max()
    else:
        norm_map = anomaly_map

    binary = (norm_map > threshold).astype(np.uint8) * 255

    # Clean up: remove tiny noise, connect close blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Image dimensions for relative size filtering
    img_area = anomaly_map.shape[0] * anomaly_map.shape[1]

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Skip noise and huge regions (> 5% of image — not a spot)
        if area < 30:
            continue
        if area > img_area * 0.05:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Skip tiny bboxes (< 5px in either dimension) — unreliable at this scale
        if w < 5 or h < 5:
            continue
        mask_region = np.zeros_like(binary)
        cv2.drawContours(mask_region, [cnt], -1, 255, -1)

        # Analyze original image in defect region
        if len(original_crop.shape) == 3:
            bgr_region = original_crop[:, :, :3]
            gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_crop

        if gray.shape[:2] != mask_region.shape[:2]:
            gray = cv2.resize(gray, (mask_region.shape[1], mask_region.shape[0]))

        defect_pixels = gray[mask_region > 0]
        mean_intensity = defect_pixels.mean() if len(defect_pixels) > 0 else 128

        # Aspect ratio determines shape
        aspect = max(w, h) / max(min(w, h), 1)

        if aspect > 3.0:
            defect_type = "scratch"
            label = "Царапина"
        elif mean_intensity < 80:
            defect_type = "spot"
            label = "Тёмное пятно/скол"
        else:
            defect_type = "spot"
            label = "Дефект покраски"

        anomaly_score = float(norm_map[mask_region > 0].mean())

        defects.append({
            "type": defect_type,
            "label": label,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(area),
            "confidence": round(min(anomaly_score * 2, 1.0), 3),
        })

    # Sort by confidence descending
    defects.sort(key=lambda d: d["confidence"], reverse=True)
    return defects
