"""
ML Pipeline: YOLO11m-seg → SAM2 → blob defect detection.
Sequential model loading to minimize VRAM usage.
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path

from app.models import (
    get_yolo, get_sam2,
    unload_model, clear_vram, get_device
)
from app.preprocessing import (
    remove_glare, crop_by_mask, suppress_glare_anomalies, classify_defects
)

logger = logging.getLogger(__name__)

# COCO class ID for 'car'
CAR_CLASS_ID = 2


def detect_car(image: np.ndarray) -> tuple[np.ndarray | None, list[int] | None, np.ndarray | None]:
    """
    Stage 1: YOLO11m-seg — detect car + coarse mask.
    Returns (mask, bbox [x1,y1,x2,y2], annotated_image) or (None, None, None).
    """
    model = get_yolo()

    results = model(image, verbose=False)[0]

    best_car = None
    best_area = 0

    if results.boxes is not None:
        # First pass: look for COCO "car" class
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if cls_id == CAR_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_car = i

        # Fallback: if no car class found, take the largest detected object
        # (handles toy cars that YOLO misclassifies)
        if best_car is None and len(results.boxes) > 0:
            logger.info("No 'car' class found, falling back to largest object")
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_car = i

    if best_car is None:
        logger.warning("No objects detected in image")
        return None, None, None

    box = results.boxes[best_car]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    bbox = [x1, y1, x2, y2]

    # Get coarse mask from YOLO segmentation
    coarse_mask = None
    if results.masks is not None and best_car < len(results.masks):
        mask_data = results.masks[best_car].data[0].cpu().numpy()
        coarse_mask = cv2.resize(
            (mask_data * 255).astype(np.uint8),
            (image.shape[1], image.shape[0])
        )

    logger.info(f"Car detected: bbox={bbox}, conf={float(box.conf[0]):.2f}")
    unload_model("yolo")

    return coarse_mask, bbox, image


def segment_car(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    """
    Stage 2: SAM2 — precise segmentation using box prompt.
    Returns binary mask (0/255).
    """
    predictor = get_sam2()

    predictor.set_image(image)

    input_box = np.array(bbox)
    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=True,
    )

    # Pick the mask with highest score
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Smooth edges with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    logger.info(f"SAM2 segmentation done, mask coverage: {mask.sum()/(mask.shape[0]*mask.shape[1])*100:.1f}%")
    unload_model("sam2")

    return mask_uint8


def _detect_wheel_zones(gray: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """
    Detect wheel areas to exclude from defect search.
    Wheels have very high local texture (spokes) in the lower portion.
    Returns binary mask where 255 = wheel zone.
    """
    h, w = gray.shape[:2]
    wheel_mask = np.zeros((h, w), dtype=np.uint8)

    # Only look in the bottom 50% of the image
    lower_start = int(h * 0.5)

    # Compute local texture via Laplacian variance in blocks
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    # Smooth to get local texture density
    texture = cv2.blur(lap, (15, 15))

    # High texture in lower body = wheel
    lower_body = body_mask.copy()
    lower_body[:lower_start, :] = False

    if lower_body.sum() < 100:
        return wheel_mask

    texture_vals = texture[lower_body]
    texture_thresh = np.percentile(texture_vals, 90)

    texture_mask = np.zeros((h, w), dtype=np.uint8)
    texture_mask[(texture > texture_thresh) & lower_body] = 255

    # Close to fill gaps between spokes, then find round-ish components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)

    # Keep only large, roughly circular blobs (wheels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(texture_mask)
    min_wheel_area = h * w * 0.01  # wheel is at least 1% of image
    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)

        # Wheel: large, roughly square (aspect < 2)
        if comp_area > min_wheel_area and aspect < 2.0:
            # Expand slightly to cover the full wheel area
            component = (labels == i).astype(np.uint8) * 255
            expanded = cv2.dilate(component, kernel, iterations=1)
            wheel_mask = cv2.bitwise_or(wheel_mask, expanded)

    return wheel_mask


def detect_defects_simple(car_crop: np.ndarray, glare_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Stage 4: Paint defect detection — finds isolated dark spots on car body.

    Approach: Blob detection + local contrast verification.
    A real paint defect (dark spot) is MUCH darker than pixels immediately around it.
    Shadows/form are gradual transitions — they fail the local contrast check.
    """
    if car_crop.shape[2] == 4:
        bgr = car_crop[:, :, :3]
        alpha = car_crop[:, :, 3]
    else:
        bgr = car_crop
        alpha = np.ones(car_crop.shape[:2], dtype=np.uint8) * 255

    body_mask = alpha > 128

    # Erode mask to exclude boundary pixels where background leaks in
    erode_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body_mask = cv2.erode(body_mask.astype(np.uint8), erode_kern).astype(bool)

    if body_mask.sum() < 100:
        return np.zeros(car_crop.shape[:2], dtype=np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    logger.info(f"Crop size: {w_img}x{h_img}")
    anomaly_map = np.zeros((h_img, w_img), dtype=np.float32)

    # ============================================================
    # STEP 0: Exclude wheel areas from body mask
    # Wheels = high texture (many edges) in the lower portion of the car
    # ============================================================
    wheel_mask = _detect_wheel_zones(gray, body_mask)
    body_mask = body_mask & (wheel_mask == 0)
    logger.info(f"Wheel mask: {wheel_mask.sum()} pixels excluded")

    # ============================================================
    # STEP 1: Blob detection — find dark circular spots
    # ============================================================
    params = cv2.SimpleBlobDetector_Params()
    # Dark blobs (darker than surroundings)
    params.filterByColor = True
    params.blobColor = 0  # dark blobs

    # Size range: from tiny chip to medium spot
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = h_img * w_img * 0.01  # max 1% of image

    # Circularity: allow irregular shapes too
    params.filterByCircularity = True
    params.minCircularity = 0.15

    # Inertia: allow elongated marks
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    # Convexity: allow irregular edges
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # Threshold steps for multi-scale detection
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector_create(params)

    # Apply body mask: set non-body pixels to body median (so no blobs there)
    body_median = int(np.median(gray[body_mask]))
    gray_masked = gray.copy()
    gray_masked[~body_mask] = body_median

    # Suppress glare before detection
    if glare_mask is not None:
        glare_resized = glare_mask
        if glare_mask.shape[:2] != gray.shape[:2]:
            glare_resized = cv2.resize(glare_mask, (w_img, h_img))
        gray_masked[glare_resized > 127] = body_median

    keypoints = list(detector.detect(gray_masked))
    logger.info(f"Blob detector found {len(keypoints)} candidates")

    # ============================================================
    # STEP 2: Adaptive threshold — catches spots blob detector misses
    # ============================================================
    # Adaptive threshold: pixel is anomalous if much darker than local mean
    blurred = cv2.GaussianBlur(gray_masked.astype(np.float32), (0, 0), sigmaX=15)
    local_contrast = blurred - gray_masked.astype(np.float32)
    # Only positive = darker than neighborhood
    local_contrast = np.clip(local_contrast, 0, None)
    local_contrast[~body_mask] = 0

    # Find strong local contrast peaks as additional candidates
    if local_contrast.max() > 0:
        lc_body = local_contrast[body_mask]
        lc_thresh = np.percentile(lc_body[lc_body > 0], 99) if (lc_body > 0).any() else 999
        lc_binary = (local_contrast > lc_thresh).astype(np.uint8)

        # Find contours of strong local contrast regions
        contours, _ = cv2.findContours(lc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < h_img * w_img * 0.005:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                # Add as keypoint candidate
                kp = cv2.KeyPoint(x=float(cx), y=float(cy), size=float(radius * 2))
                keypoints.append(kp)

    # ============================================================
    # STEP 2b: Dark patch detection — finds irregular (non-round) dark marks
    # Threshold-based: anything significantly darker than body average
    # ============================================================
    dark_thresh = body_median * 0.70  # 30% darker than body average
    dark_binary = np.zeros_like(gray, dtype=np.uint8)
    dark_binary[(gray < dark_thresh) & body_mask] = 255

    # Clean tiny noise
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, kern_open)

    dark_contours, _ = cv2.findContours(dark_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_patch_count = 0
    for cnt in dark_contours:
        area = cv2.contourArea(cnt)
        if 15 < area < h_img * w_img * 0.01:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            kp = cv2.KeyPoint(x=float(cx), y=float(cy), size=float(max(radius, 3) * 2))
            keypoints.append(kp)
            dark_patch_count += 1

    logger.info(f"Total candidates: {len(keypoints)} (blob + adaptive + {dark_patch_count} dark patches)")

    # ============================================================
    # STEP 3: Verify each candidate — local contrast ratio check
    # A real defect has HIGH contrast vs its immediate ring neighborhood
    # ============================================================
    verified = []
    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = max(int(kp.size / 2), 3)

        # Skip if outside body
        if not body_mask[min(cy, h_img-1), min(cx, w_img-1)]:
            continue

        # Inner circle (the spot itself)
        inner_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), r, 255, -1)
        inner_mask &= (body_mask.astype(np.uint8) * 255)

        # Outer ring (neighborhood around the spot)
        outer_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(outer_mask, (cx, cy), r * 3, 255, -1)
        cv2.circle(outer_mask, (cx, cy), r + 2, 0, -1)
        outer_mask &= (body_mask.astype(np.uint8) * 255)

        inner_pixels = gray[inner_mask > 0]
        outer_pixels = gray[outer_mask > 0]

        if len(inner_pixels) < 5 or len(outer_pixels) < 10:
            continue

        inner_mean = inner_pixels.mean()
        outer_mean = outer_pixels.mean()
        outer_std = outer_pixels.std()

        # The spot must be significantly darker than its ring
        contrast = outer_mean - inner_mean

        # Relative contrast: how dark is the spot compared to neighborhood
        rel_contrast = contrast / max(outer_mean, 1)

        # Smoothness check: surrounding area must be smooth (low std = uniform paint)
        # Smooth body panel: std ~3-8. Wheel/textured area: std > 12
        is_smooth_area = outer_std < 12

        # Statistical significance: contrast must exceed 2.5× local noise
        # On smooth paint (std~5): need contrast > 12.5 — easy for real defects
        # On variable area (std~8): need contrast > 20 — filters natural variation
        stat_significant = contrast > 2.5 * outer_std

        logger.info(
            f"  Candidate ({cx},{cy}) r={r}: inner={inner_mean:.0f} outer={outer_mean:.0f} "
            f"contrast={contrast:.0f} rel={rel_contrast:.2f} outer_std={outer_std:.1f} "
            f"smooth={is_smooth_area} stat_sig={stat_significant}"
        )

        # Must be: darker than surroundings AND surroundings are smooth paint
        # AND contrast is statistically significant (not just natural variation)
        if not (rel_contrast > 0.20 and contrast > 12 and is_smooth_area and stat_significant):
            continue

        # Extra wheel check: candidates in the bottom 35% of crop are in wheel zone.
        # Require much stronger evidence (higher contrast) to accept.
        # Wheel false positives: contrast ~15-20, rel ~0.25-0.35
        # Real defects: contrast ~25+, rel ~0.40+
        is_wheel = False
        if cy > h_img * 0.65:
            if contrast < 25 or rel_contrast < 0.40:
                is_wheel = True
                logger.info(f"  Rejected ({cx},{cy}): weak candidate in wheel zone")

        if not is_wheel:
            score = min(rel_contrast * 2, 1.0)
            verified.append((cx, cy, r, score))
            # Paint into anomaly map
            cv2.circle(anomaly_map, (cx, cy), r, float(score), -1)

    # ============================================================
    # STEP 4: Remove clusters — if 3+ spots within 50px, it's a
    # structural feature (wheel arch, grille), not individual defects
    # ============================================================
    if len(verified) >= 3:
        cluster_radius = 60
        to_remove = set()
        for i, (x1, y1, r1, s1) in enumerate(verified):
            neighbors = 0
            for j, (x2, y2, r2, s2) in enumerate(verified):
                if i != j:
                    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if dist < cluster_radius:
                        neighbors += 1
            if neighbors >= 2:  # this point has 2+ neighbors nearby = cluster
                to_remove.add(i)

        if to_remove:
            logger.info(f"Removing {len(to_remove)} clustered false positives")
            # Clear their circles from anomaly_map
            for idx in to_remove:
                cx, cy, r, _ = verified[idx]
                cv2.circle(anomaly_map, (cx, cy), r + 2, 0, -1)
            verified = [v for i, v in enumerate(verified) if i not in to_remove]

    logger.info(f"Final defects: {len(verified)}")
    return anomaly_map


def create_heatmap_overlay(image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    """Draw defect markers on image. Only circles where anomaly_map > 0."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        vis = image[:, :, :3].copy()
    else:
        vis = image.copy()

    if anomaly_map.max() < 0.01:
        return vis

    heatmap_resized = cv2.resize(anomaly_map, (vis.shape[1], vis.shape[0]))

    # Find defect blobs and draw circles
    contours, _ = cv2.findContours(
        (heatmap_resized > 0.01).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        r = max(int(radius) + 5, 10)
        cv2.circle(vis, (int(cx), int(cy)), r, (0, 0, 255), 2)
        cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    return vis


def process_image(image_path: str, result_dir: str, progress_callback=None) -> dict:
    """
    Full pipeline: detect car → segment → remove glare → detect defects.
    Saves results to result_dir and returns metadata.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Failed to load image"}

    # Resize large images to prevent VRAM overflow (RTX 4050 6GB limit)
    MAX_SIDE = 2048
    h, w = image.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized {w}x{h} -> {new_w}x{new_h} (scale {scale:.2f})")

    logger.info(f"Processing {image_path}, size: {image.shape}")

    # Save original
    cv2.imwrite(str(result_path / "original.jpg"), image)

    # Stage 1: YOLO detection
    _progress("Этап 1: Обнаружение авто (YOLO)")
    logger.info("Stage 1: Car detection (YOLO11m-seg)")
    coarse_mask, bbox, _ = detect_car(image)

    if bbox is None:
        return {
            "error": "No car detected in the image",
            "original": "original.jpg",
        }

    # Stage 2: SAM2 precise segmentation
    _progress("Этап 2: Сегментация (SAM2)")
    logger.info("Stage 2: Precise segmentation (SAM2)")
    try:
        precise_mask = segment_car(image, bbox)
    except Exception as e:
        logger.warning(f"SAM2 failed, using YOLO mask: {e}")
        if coarse_mask is not None:
            precise_mask = coarse_mask
        else:
            # Fallback: use bbox as mask
            precise_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            precise_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    # Save mask
    cv2.imwrite(str(result_path / "mask.png"), precise_mask)

    # Create masked car visualization
    car_vis = image.copy()
    car_vis[precise_mask < 128] = 0
    cv2.imwrite(str(result_path / "car_segmented.jpg"), car_vis)

    # Stage 3: Crop + preprocess
    _progress("Этап 3: Предобработка")
    logger.info("Stage 3: Crop + glare removal")
    car_crop, crop_bbox = crop_by_mask(image, precise_mask)

    # Remove glare from crop
    crop_bgr = car_crop[:, :, :3] if car_crop.shape[2] == 4 else car_crop
    cleaned_crop, glare_mask = remove_glare(crop_bgr)
    cv2.imwrite(str(result_path / "car_crop.png"), car_crop)

    # Stage 4: Defect detection (blob detector + local contrast)
    _progress("Этап 4: Поиск дефектов")
    logger.info("Stage 4: Defect detection")
    anomaly_map = detect_defects_simple(car_crop, glare_mask)

    # Stage 5: Post-processing
    _progress("Этап 5: Классификация дефектов")
    logger.info("Stage 5: Post-processing + classification")
    defects = classify_defects(anomaly_map, car_crop, threshold=0.01)

    # Rebuild anomaly map from filtered defects only (remove false positives)
    filtered_map = np.zeros_like(anomaly_map)
    for d in defects:
        dx, dy, dw, dh = d["bbox"]
        cx = dx + dw // 2
        cy = dy + dh // 2
        r = max(dw, dh) // 2 + 1
        cv2.circle(filtered_map, (cx, cy), r, float(d["confidence"]), -1)

    # Create heatmap overlay using filtered map (consistent with annotations)
    heatmap_overlay = create_heatmap_overlay(car_crop, filtered_map)
    cv2.imwrite(str(result_path / "heatmap.jpg"), heatmap_overlay)

    # Create full-image annotated version
    annotated = image.copy()

    # Draw defect markers on full image
    ox, oy = crop_bbox[0], crop_bbox[1]
    for i, d in enumerate(defects):
        dx, dy, dw, dh = d["bbox"]
        cx = ox + dx + dw // 2
        cy = oy + dy + dh // 2
        r = max(int(max(dw, dh) * 0.7) + 5, 10)

        # Red circle around defect + center dot
        cv2.circle(annotated, (cx, cy), r, (0, 0, 255), 2)
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)

        # Label in English (cv2 can't render Cyrillic)
        cv2.putText(annotated, f"#{i+1}", (cx + r + 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(str(result_path / "annotated.jpg"), annotated)

    # Save anomaly map as grayscale
    cv2.imwrite(str(result_path / "anomaly_map.png"),
                (anomaly_map * 255).astype(np.uint8))

    # Unload all models
    clear_vram()

    result = {
        "original": "original.jpg",
        "mask": "mask.png",
        "car_segmented": "car_segmented.jpg",
        "car_crop": "car_crop.png",
        "heatmap": "heatmap.jpg",
        "annotated": "annotated.jpg",
        "anomaly_map": "anomaly_map.png",
        "bbox": bbox,
        "crop_offset": [crop_bbox[0], crop_bbox[1]],
        "defects": defects,
        "defect_count": len(defects),
    }

    logger.info(f"Done. Found {len(defects)} defects.")
    return result
