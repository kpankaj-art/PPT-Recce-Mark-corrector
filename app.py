
import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
import shutil
import gc
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


# ============================================================
# APP SETTINGS
# ============================================================

st.set_page_config(
    page_title="PPT Recce Mark Corrector",
    page_icon="🟩",
    layout="wide"
)

MAX_PPT_MB = 300
MAX_IMAGE_PIXELS = 40_000_000
CHUNK_SIZE = 1024 * 1024

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# OpenCV works locally; no API key required.


# ============================================================
# MEMORY
# ============================================================

def clean_memory():
    gc.collect()


# ============================================================
# PPT MEDIA REFERENCES
# ============================================================

def get_slide_media_references(ppt_path):
    """
    Finds raster images actually referenced by slides.
    Does not load the whole PPT into memory.
    """
    media_paths = set()

    with zipfile.ZipFile(ppt_path, "r") as z:
        names = set(z.namelist())

        for rels_name in names:
            if not rels_name.startswith("ppt/slides/_rels/"):
                continue
            if not rels_name.endswith(".rels"):
                continue

            try:
                root = ET.fromstring(z.read(rels_name))
            except Exception:
                continue

            rel_map = {}

            for rel in root:
                rid = rel.attrib.get("Id")
                target = rel.attrib.get("Target")

                if not rid or not target:
                    continue

                # Handle ../media/image1.png
                if "../media/" in target:
                    filename = target.split("../media/", 1)[1]
                    rel_map[rid] = "ppt/media/" + filename

            rel_filename = Path(rels_name).name
            slide_filename = rel_filename[:-5]
            slide_xml = "ppt/slides/" + slide_filename

            if slide_xml not in names:
                continue

            try:
                slide_root = ET.fromstring(z.read(slide_xml))
            except Exception:
                continue

            for element in slide_root.iter():
                for attr_name, attr_value in element.attrib.items():
                    if attr_name.endswith("}embed"):
                        if attr_value in rel_map:
                            media_paths.add(rel_map[attr_value])

    return sorted(media_paths)


# ============================================================
# IMAGE HELPERS
# ============================================================

def read_image_from_file(path):
    """
    Reads one image only.
    Keeps alpha if PNG has alpha.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Image could not be read.")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def save_image(path, img):
    ext = Path(path).suffix.lower()

    params = []

    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 94]

    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 94]

    ok = cv2.imwrite(path, img, params)

    if not ok:
        raise ValueError("Could not save processed image.")


def bgr_for_processing(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        return img[:, :, :3]

    return img


def restore_alpha(original, processed):
    if original.ndim == 3 and original.shape[2] == 4:
        alpha = original[:, :, 3]
        if processed.ndim == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGRA)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)
        processed[:, :, 3] = alpha
        return processed

    return processed


# ============================================================
# DETECTION
# ============================================================

def clean_mask(mask):
    """
    Cleans isolated noise while keeping a hand-drawn line.
    """
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_close, iterations=1
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_open, iterations=1
    )

    return mask


def component_candidates(mask, image_shape):
    """
    Returns large plausible marking components.
    """
    h, w = image_shape[:2]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates = []

    for i in range(1, n):
        x, y, cw, ch, area = stats[i]

        if area < 500:
            continue

        if cw < 50 or ch < 25:
            continue

        # Very large full-image components are usually photo content.
        if cw > w * 0.90 and ch > h * 0.75:
            continue

        # Bottom GPS/location overlay is normally not the marking.
        if y > h * 0.82:
            continue

        bbox_area = cw * ch
        fill = area / max(1, bbox_area)

        # Hand-drawn outline should not fill the entire bbox.
        if fill > 0.55:
            continue

        roi = labels[y:y + ch, x:x + cw] == i

        border = max(
            5,
            min(14, int(min(cw, ch) * 0.05))
        )

        border_band = np.zeros_like(roi, dtype=np.uint8)
        border_band[:border, :] = 1
        border_band[-border:, :] = 1
        border_band[:, :border] = 1
        border_band[:, -border:] = 1

        border_density = (
            (roi & (border_band > 0)).sum()
            / max(1, roi.sum())
        )

        # Prefer components with pixels close to the bbox border.
        score = (
            min(area / 10000.0, 10.0)
            + border_density * 20.0
        )

        candidates.append({
            "label": i,
            "x": int(x),
            "y": int(y),
            "w": int(cw),
            "h": int(ch),
            "area": int(area),
            "fill": float(fill),
            "border_density": float(border_density),
            "score": float(score),
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    return candidates, labels


def green_detection(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Bright/saturated green.
    mask = cv2.inRange(
        hsv,
        np.array([35, 70, 60], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8)
    )

    mask = clean_mask(mask)

    candidates, labels = component_candidates(
        mask, bgr.shape
    )

    if not candidates:
        return None

    # Green marking in the sample is a dominant large component.
    best = candidates[0]

    # Require reasonable size.
    if best["w"] * best["h"] < bgr.shape[0] * bgr.shape[1] * 0.01:
        return None

    component = (
        labels == best["label"]
    ).astype(np.uint8) * 255

    return {
        "mask": component,
        "bbox": (
            best["x"],
            best["y"],
            best["w"],
            best["h"]
        ),
        "method": "green",
        "confidence": min(
            0.99,
            0.50
            + min(best["border_density"], 0.20)
            + min(best["area"] /
                  (bgr.shape[0] * bgr.shape[1]) * 2.0, 0.30)
        )
    }


def black_detection(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Dark ink / black marker.
    mask = cv2.inRange(
        gray,
        0,
        78
    )

    # Connect thick hand-drawn strokes.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9)
        ),
        iterations=1
    )

    candidates, labels = component_candidates(
        mask, bgr.shape
    )

    if not candidates:
        return None

    # Additional scoring for a rectangle-like outline.
    scored = []

    H, W = bgr.shape[:2]

    for c in candidates:
        x, y, w, h = (
            c["x"], c["y"], c["w"], c["h"]
        )

        aspect = w / max(1, h)

        # Most recce markings are not extremely thin.
        if aspect < 1.15 or aspect > 8.0:
            continue

        area_ratio = (
            (w * h) /
            max(1, W * H)
        )

        if area_ratio < 0.01:
            continue

        score = c["score"]

        # Strongly prefer outline-like components.
        score += c["border_density"] * 25

        # Prefer moderate fill.
        if c["fill"] < 0.20:
            score += 2.0

        if 0.02 < area_ratio < 0.45:
            score += 1.0

        scored.append(
            (score, c)
        )

    if not scored:
        return None

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best = scored[0][1]

    component = (
        labels == best["label"]
    ).astype(np.uint8) * 255

    return {
        "mask": component,
        "bbox": (
            best["x"],
            best["y"],
            best["w"],
            best["h"]
        ),
        "method": "black",
        "confidence": min(
            0.98,
            0.45
            + min(best["border_density"], 0.25)
            + (0.15 if best["fill"] < 0.20 else 0.0)
        )
    }


# ============================================================
# REMOVE ONLY THE MARKER LINE
# ============================================================

def make_line_only_mask(component_mask, bbox):
    """
    Important:
    Do NOT erase the whole component because the target can
    contain real dark text/details.

    Only erase the portion of the detected component that lies
    close to the outer bounding-box border.
    """
    x, y, w, h = bbox

    roi = component_mask[
        y:y + h,
        x:x + w
    ]

    band = max(
        7,
        min(18, int(min(w, h) * 0.06))
    )

    border_band = np.zeros_like(
        roi,
        dtype=np.uint8
    )

    border_band[:band, :] = 255
    border_band[-band:, :] = 255
    border_band[:, :band] = 255
    border_band[:, -band:] = 255

    line_mask_roi = cv2.bitwise_and(
        roi,
        border_band
    )

    # Put back into full image coordinates.
    full = np.zeros_like(
        component_mask
    )

    full[
        y:y + h,
        x:x + w
    ] = line_mask_roi

    # Slight expansion removes antialiased marker edges.
    full = cv2.dilate(
        full,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        ),
        iterations=1
    )

    return full


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    input_path,
    output_path,
    mode="Auto",
    rectangle_thickness=4,
    inpaint_radius=4
):

    original = read_image_from_file(
        input_path
    )

    if original.shape[0] * original.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Image is too large for safe processing."
        )

    bgr = bgr_for_processing(
        original
    )

    detection = None

    if mode in ("Auto", "Green"):
        detection = green_detection(
            bgr
        )

    if detection is None and mode in (
        "Auto",
        "Black"
    ):
        detection = black_detection(
            bgr
        )

    if detection is None:
        # No reliable marking detected.
        # Copy original image unchanged.
        shutil.copyfile(
            input_path,
            output_path
        )

        clean_memory()

        return {
            "changed": False,
            "method": "none",
            "confidence": 0.0,
            "bbox": None
        }

    # ================================================
    # Build mask that removes only outer marker line
    # ================================================

    line_mask = make_line_only_mask(
        detection["mask"],
        detection["bbox"]
    )

    # ================================================
    # Inpaint
    # ================================================

    repaired = cv2.inpaint(
        bgr,
        line_mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA
    )

    # ================================================
    # Draw professional rectangle
    # ================================================

    x, y, w, h = detection["bbox"]

    H, W = repaired.shape[:2]

    # Keep rectangle safely inside image.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W - 1, x + w - 1)
    y2 = min(H - 1, y + h - 1)

    cv2.rectangle(
        repaired,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        thickness=rectangle_thickness,
        lineType=cv2.LINE_AA
    )

    # Restore alpha if original PNG has transparency.
    result = restore_alpha(
        original,
        repaired
    )

    save_image(
        output_path,
        result
    )

    del original
    del bgr
    del repaired
    del result
    del line_mask
    del detection

    clean_memory()

    return {
        "changed": True,
        "method": detection["method"],
        "confidence": detection["confidence"],
        "bbox": (
            int(x),
            int(y),
            int(w),
            int(h)
        )
    }


# ============================================================
# EXTRACT ONE MEDIA FILE
# ============================================================

def extract_media(
    ppt_zip,
    media_path,
    destination
):

    with ppt_zip.open(
        media_path,
        "r"
    ) as src:

        with open(
            destination,
            "wb"
        ) as dst:

            while True:
                chunk = src.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                dst.write(
                    chunk
                )


# ============================================================
# BUILD FINAL PPT
# ============================================================

def build_final_ppt(
    original_ppt,
    output_ppt,
    replacements
):

    with zipfile.ZipFile(
        original_ppt,
        "r"
    ) as zin:

        with zipfile.ZipFile(
            output_ppt,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as zout:

            for info in zin.infolist():

                filename = info.filename

                replacement = replacements.get(
                    filename
                )

                if (
                    replacement
                    and
                    os.path.exists(
                        replacement
                    )
                ):

                    new_info = zipfile.ZipInfo(
                        filename
                    )

                    new_info.date_time = (
                        info.date_time
                    )

                    new_info.compress_type = (
                        zipfile.ZIP_DEFLATED
                    )

                    with open(
                        replacement,
                        "rb"
                    ) as src:

                        with zout.open(
                            new_info,
                            "w"
                        ) as dst:

                            while True:
                                chunk = src.read(
                                    CHUNK_SIZE
                                )

                                if not chunk:
                                    break

                                dst.write(
                                    chunk
                                )

                else:

                    with zin.open(
                        info,
                        "r"
                    ) as src:

                        with zout.open(
                            info,
                            "w"
                        ) as dst:

                            while True:
                                chunk = src.read(
                                    CHUNK_SIZE
                                )

                                if not chunk:
                                    break

                                dst.write(
                                    chunk
                                )

    clean_memory()


# ============================================================
# UI
# ============================================================

st.title(
    "🖼️ PPT Recce Mark Corrector"
)

st.write(
    "Baked-in rough/hand-drawn marking ko "
    "OpenCV se remove karke clean professional "
    "rectangle banaye."
)

st.success(
    "✅ No Gemini API • No OpenAI API • No API cost"
)

st.info(
    "Processing local/OpenCV based hai. "
    "800+ slides ke liye images one-by-one process hongi."
)


# ============================================================
# OPTIONS
# ============================================================

with st.expander(
    "⚙️ Detection Settings",
    expanded=True
):

    detection_mode = st.selectbox(
        "Marking colour",
        [
            "Auto",
            "Green",
            "Black"
        ],
        index=0
    )

    col1, col2 = st.columns(2)

    with col1:

        rectangle_thickness = st.slider(
            "Professional rectangle thickness",
            min_value=1,
            max_value=12,
            value=4
        )

    with col2:

        inpaint_radius = st.slider(
            "Background repair strength",
            min_value=1,
            max_value=8,
            value=4
        )

st.subheader(
    "📂 Upload PowerPoint"
)

uploaded_file = st.file_uploader(
    "PPTX upload karo — maximum 300 MB",
    type=["pptx"]
)


if uploaded_file:

    size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.write(
        f"📦 File size: **{size_mb:.2f} MB**"
    )

    if size_mb > MAX_PPT_MB:
        st.error(
            "❌ PPT 300 MB se badi hai."
        )
        st.stop()

    if st.button(
        "🚀 Start Processing",
        type="primary",
        use_container_width=True
    ):

        work_dir = tempfile.mkdtemp(
            prefix="ppt_mark_"
        )

        input_ppt = os.path.join(
            work_dir,
            "input.pptx"
        )

        output_ppt = os.path.join(
            work_dir,
            "Professional_Marked.pptx"
        )

        try:

            # ==========================================
            # SAVE PPT
            # ==========================================

            with st.status(
                "📥 PPT save ho rahi hai...",
                expanded=True
            ) as status:

                save_uploaded = open(
                    input_ppt,
                    "wb"
                )

                try:
                    while True:
                        chunk = uploaded_file.read(
                            CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        save_uploaded.write(
                            chunk
                        )
                finally:
                    save_uploaded.close()

                status.update(
                    label="✅ PPT save ho gayi",
                    state="complete"
                )

            # ==========================================
            # FIND IMAGES
            # ==========================================

            with st.status(
                "🔎 Slides ki images identify ho rahi hain...",
                expanded=True
            ) as status:

                media_paths = (
                    get_slide_media_references(
                        input_ppt
                    )
                )

                raster_paths = [
                    p for p in media_paths
                    if Path(p).suffix.lower()
                    in SUPPORTED
                ]

                status.write(
                    f"Unique slide images: "
                    f"**{len(raster_paths)}**"
                )

                status.update(
                    label="✅ Images identify ho gayi",
                    state="complete"
                )

            if not raster_paths:
                st.warning(
                    "PPT me supported JPG/PNG/WebP/BMP "
                    "images nahi mili."
                )
                st.stop()

            # ==========================================
            # PROCESS
            # ==========================================

            st.subheader(
                "⚙️ Image Processing"
            )

            progress = st.progress(
                0
            )

            current = st.empty()

            replacements = {}

            changed = 0
            unchanged = 0
            failed = 0

            with zipfile.ZipFile(
                input_ppt,
                "r"
            ) as ppt_zip:

                total = len(
                    raster_paths
                )

                for number, media_path in enumerate(
                    raster_paths,
                    start=1
                ):

                    name = Path(
                        media_path
                    ).name

                    current.write(
                        f"🖼️ {number}/{total} — {name}"
                    )

                    source_path = os.path.join(
                        work_dir,
                        "src_" + name
                    )

                    output_path = os.path.join(
                        work_dir,
                        "out_" + name
                    )

                    try:

                        extract_media(
                            ppt_zip,
                            media_path,
                            source_path
                        )

                        result = process_image(
                            source_path,
                            output_path,
                            mode=detection_mode,
                            rectangle_thickness=(
                                rectangle_thickness
                            ),
                            inpaint_radius=(
                                inpaint_radius
                            )
                        )

                        if result["changed"]:

                            replacements[
                                media_path
                            ] = output_path

                            changed += 1

                        else:

                            unchanged += 1

                    except Exception as e:

                        failed += 1

                        st.warning(
                            f"⚠️ {name} process nahi hua: "
                            f"{e}"
                        )

                    finally:

                        # Source can be deleted.
                        # Replacement stays on disk until PPT build.
                        try:
                            if os.path.exists(
                                source_path
                            ):
                                os.remove(
                                    source_path
                                )
                        except Exception:
                            pass

                        clean_memory()

                    progress.progress(
                        number / total
                    )

            current.write(
                "✅ Image processing complete"
            )

            # ==========================================
            # BUILD PPT
            # ==========================================

            with st.status(
                "📦 Final PPT ban rahi hai...",
                expanded=True
            ) as status:

                build_final_ppt(
                    input_ppt,
                    output_ppt,
                    replacements
                )

                status.update(
                    label="✅ Final PPT ready",
                    state="complete"
                )

            final_mb = (
                os.path.getsize(output_ppt)
                /
                (1024 * 1024)
            )

            st.success(
                "🎉 Processing complete!"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Images found",
                    len(raster_paths)
                )

            with c2:
                st.metric(
                    "Markings fixed",
                    changed
                )

            with c3:
                st.metric(
                    "No marking",
                    unchanged
                )

            with c4:
                st.metric(
                    "Failed",
                    failed
                )

            st.write(
                f"📦 Final PPT size: **{final_mb:.2f} MB**"
            )

            with open(
                output_ppt,
                "rb"
            ) as download:

                st.download_button(
                    "⬇️ Download Professional PPT",
                    data=download,
                    file_name=(
                        Path(
                            uploaded_file.name
                        ).stem
                        +
                        "_Professional.pptx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.presentationml.presentation"
                    ),
                    use_container_width=True
                )

            st.info(
                "Slide XML/layout ko modify nahi kiya gaya. "
                "Sirf processed raster image files replace hui hain."
            )

        except Exception as e:

            st.error(
                "❌ Processing error"
            )

            st.exception(e)

        finally:

            # Do not delete work_dir immediately because
            # Streamlit's download button may still need the file.
            clean_memory()
