
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFile
import zipfile
import tempfile
import shutil
import gc
import os
from pathlib import Path
import xml.etree.ElementTree as ET

st.set_page_config(
    page_title="PPT Recce Mark Corrector V2",
    page_icon="🔲",
    layout="wide"
)

MAX_PPT_MB = 300
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# MARKING DETECTION
# ============================================================

def green_marker_mask(bgr):
    """Detect common green survey/highlighter markings."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad green range; avoids most natural blue/red markings.
    lower = np.array([30, 55, 45], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Join broken marker strokes.
    k = max(3, int(min(bgr.shape[:2]) * 0.006) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove tiny color noise.
    small = max(20, int(bgr.shape[0] * bgr.shape[1] * 0.00001))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= small:
            clean[labels == i] = 255
    return clean


def black_marker_mask(bgr):
    """
    Detect a dark rough outline rather than blindly removing every
    black pixel. Candidate dark pixels are connected/closed and then
    scored by size and outline-like geometry.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive dark threshold: avoids being too aggressive on normal photos.
    # Keep very dark pixels, but reject almost-white compression noise.
    dark = cv2.inRange(gray, 0, 58)

    # Connect thick/rough strokes without destroying the whole image.
    k1 = max(3, int(min(h, w) * 0.004) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate slightly so anti-aliased edges of the old marker are removed.
    kd = max(3, int(min(h, w) * 0.0025) | 1)
    kd_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    closed = cv2.dilate(closed, kd_kernel, iterations=1)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = h * w
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        box_area = bw * bh

        if box_area < image_area * 0.015:
            continue
        if box_area > image_area * 0.90:
            continue
        if bw < w * 0.08 or bh < h * 0.08:
            continue

        # Prefer large, non-border objects.
        border_touch = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2

        rect_area = float(max(box_area, 1))
        fill_ratio = area / rect_area

        # Rough outlines usually have relatively low fill ratio.
        # Solid dark objects/text blocks have much higher fill ratio.
        score = 0.0
        score += min(box_area / image_area, 0.80) * 2.0
        score += max(0.0, 0.35 - fill_ratio) * 2.0
        score -= 0.8 if border_touch else 0.0

        # Prefer landscape/portrait rectangles rather than tiny squares.
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 12:
            score -= 0.5

        candidates.append((score, c, (x, y, bw, bh)))

    if not candidates:
        return np.zeros_like(gray), None

    candidates.sort(key=lambda z: z[0], reverse=True)
    _, contour, bbox = candidates[0]

    # IMPORTANT: Do not erase the whole bounding box.
    # Only remove dark pixels close to the detected rough contour.
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    x, y, bw, bh = bbox
    pad = max(4, int(min(h, w) * 0.018))
    outer = np.zeros_like(gray)
    cv2.rectangle(
        outer,
        (max(0, x - pad), max(0, y - pad)),
        (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)),
        255,
        thickness=-1
    )

    # Ring around contour/bounding perimeter.
    eroded = cv2.erode(
        contour_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, pad // 2 * 2 + 1), max(3, pad // 2 * 2 + 1))
        ),
        iterations=1
    )
    ring = cv2.subtract(contour_mask, eroded)
    ring = cv2.bitwise_and(ring, outer)

    # Only dark pixels in this ring become removal mask.
    final_mask = cv2.bitwise_and(dark, ring)

    # Expand a little to catch anti-aliased marker edges.
    expand = max(3, int(min(h, w) * 0.004) | 1)
    ex_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand, expand)
    )
    final_mask = cv2.dilate(final_mask, ex_kernel, iterations=1)

    return final_mask, bbox


def detect_marker(bgr, mode="Auto"):
    green = green_marker_mask(bgr)
    black, black_bbox = black_marker_mask(bgr)

    green_area = int(np.count_nonzero(green))
    black_area = int(np.count_nonzero(black))

    if mode == "Green":
        chosen = green
        color = "green"
    elif mode == "Black":
        chosen = black
        color = "black"
    else:
        # Prefer green when a substantial green stroke exists.
        total = bgr.shape[0] * bgr.shape[1]
        if green_area > max(100, total * 0.00015):
            chosen = green
            color = "green"
        elif black_area > max(100, total * 0.00008):
            chosen = black
            color = "black"
        else:
            chosen = np.zeros_like(green)
            color = None

    if np.count_nonzero(chosen) < 30:
        return np.zeros_like(green), None, None

    # Determine target box from the marker mask itself.
    ys, xs = np.where(chosen > 0)
    if len(xs) == 0:
        return np.zeros_like(green), None, None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return chosen, (x1, y1, x2, y2), color


# ============================================================
# REMOVE OLD MARKING + DRAW NEW RECTANGLE
# ============================================================

def repair_and_mark(pil_image, mode="Auto", line_width_ratio=0.004):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mask, bbox, detected_color = detect_marker(bgr, mode)

    if bbox is None:
        return pil_image.convert("RGB"), None, None

    # Remove the OLD marker completely.
    # Larger inpainting radius is used for thick hand-drawn strokes.
    radius = max(3, int(min(bgr.shape[:2]) * 0.008))

    repaired = cv2.inpaint(
        bgr,
        mask,
        radius,
        cv2.INPAINT_TELEA
    )

    x1, y1, x2, y2 = bbox

    # Small expansion so the professional rectangle encloses the same target,
    # not the marker pixels themselves.
    h, w = repaired.shape[:2]
    pad_x = max(2, int((x2 - x1) * 0.01))
    pad_y = max(2, int((y2 - y1) * 0.01))

    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w - 1, x2 + pad_x)
    ry2 = min(h - 1, y2 + pad_y)

    thickness = max(
        2,
        int(min(h, w) * line_width_ratio)
    )

    # Draw clean professional rectangle.
    cv2.rectangle(
        repaired,
        (rx1, ry1),
        (rx2, ry2),
        (0, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    result_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result_rgb)

    # Preserve original dimensions.
    if result.size != pil_image.size:
        result = result.resize(
            pil_image.size,
            Image.Resampling.LANCZOS
        )

    del rgb, bgr, mask, repaired, result_rgb
    gc.collect()

    return result, bbox, detected_color


# ============================================================
# PPT ZIP IMAGE DISCOVERY
# ============================================================

def get_slide_media_references(ppt_path):
    media = set()

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

            relmap = {}

            for rel in root:
                rid = rel.attrib.get("Id")
                target = rel.attrib.get("Target", "")
                if rid and "../media/" in target:
                    media_name = target.split("../media/")[-1]
                    relmap[rid] = "ppt/media/" + media_name

            slide_name = Path(rels_name).name[:-5]
            slide_xml = "ppt/slides/" + slide_name

            if slide_xml not in names:
                continue

            try:
                slide_root = ET.fromstring(z.read(slide_xml))
            except Exception:
                continue

            for element in slide_root.iter():
                for attr, value in element.attrib.items():
                    if attr.endswith("}embed") and value in relmap:
                        media.add(relmap[value])

    return sorted(media)


# ============================================================
# PROCESS ONE MEDIA FILE
# ============================================================

def process_media(ppt_zip, media_path, work_dir, mode):
    ext = Path(media_path).suffix.lower()

    if ext not in SUPPORTED:
        return False, "unsupported", None

    safe = media_path.replace("/", "_").replace("\\", "_")

    original_path = os.path.join(work_dir, "orig_" + safe)
    output_path = os.path.join(work_dir, "fixed_" + safe + ".jpg")

    try:
        with ppt_zip.open(media_path, "r") as src, open(original_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        with Image.open(original_path) as img:
            img.load()
            original = img.convert("RGB")

        result, bbox, color = repair_and_mark(original, mode)

        if bbox is None:
            # Keep image unchanged when no marker is confidently detected.
            shutil.copyfile(original_path, output_path)
            original.close()
            result.close()
            return True, "no_marker", output_path

        result.save(
            output_path,
            format="JPEG",
            quality=94,
            optimize=False
        )

        original.close()
        result.close()

        try:
            os.remove(original_path)
        except Exception:
            pass

        gc.collect()

        return True, "fixed_" + str(color), output_path

    except Exception as e:
        for p in [original_path, output_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        gc.collect()
        return False, str(e), None


# ============================================================
# BUILD FINAL PPT
# ============================================================

def build_final_ppt(input_ppt, output_ppt, replacements):
    with zipfile.ZipFile(input_ppt, "r") as zin:
        with zipfile.ZipFile(
            output_ppt,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as zout:

            for info in zin.infolist():
                replacement = replacements.get(info.filename)

                if replacement and os.path.exists(replacement):
                    new_info = zipfile.ZipInfo(info.filename)
                    new_info.date_time = info.date_time
                    new_info.compress_type = zipfile.ZIP_DEFLATED

                    with open(replacement, "rb") as src:
                        with zout.open(new_info, "w") as dst:
                            while True:
                                chunk = src.read(1024 * 1024)
                                if not chunk:
                                    break
                                dst.write(chunk)
                else:
                    with zin.open(info, "r") as src:
                        with zout.open(info, "w") as dst:
                            while True:
                                chunk = src.read(1024 * 1024)
                                if not chunk:
                                    break
                                dst.write(chunk)

    gc.collect()


# ============================================================
# UI
# ============================================================

st.title("🔲 PPT Recce Mark Corrector V2")

st.write(
    "Baked-in green/black rough marking ko remove karke "
    "same target area par clean straight professional rectangle banata hai."
)

st.info(
    "Local/OpenCV version — Gemini/OpenAI API ki zarurat nahi."
)

st.subheader("🎯 Marking Type")

mode = st.selectbox(
    "Photo me purani marking kis type ki hai?",
    ["Auto", "Green", "Black"],
    index=0
)

st.caption(
    "Auto green ko priority deta hai. Black mode sirf dark rough-outline "
    "candidate ko target karta hai."
)

st.subheader("📂 Upload PPTX")

uploaded = st.file_uploader(
    "Maximum 300 MB",
    type=["pptx"]
)

if uploaded:
    size_mb = uploaded.size / (1024 * 1024)
    st.write(f"📦 Size: **{size_mb:.2f} MB**")

    if size_mb > MAX_PPT_MB:
        st.error("❌ PPT 300 MB se badi hai.")
        st.stop()

    if st.button(
        "🚀 Start Processing",
        type="primary",
        use_container_width=True
    ):
        work_dir = tempfile.mkdtemp(prefix="ppt_recc_v2_")
        input_ppt = os.path.join(work_dir, "input.pptx")
        output_ppt = os.path.join(work_dir, "output_Professional.pptx")

        try:
            with st.status("📥 PPT save ho rahi hai...") as s:
                with open(input_ppt, "wb") as f:
                    while True:
                        chunk = uploaded.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                s.update(label="✅ PPT save ho gayi", state="complete")

            with st.status("🔎 Images identify ho rahi hain...") as s:
                media_paths = get_slide_media_references(input_ppt)
                s.write(f"**{len(media_paths)}** unique images found")
                s.update(label="✅ Images identify ho gayi", state="complete")

            if not media_paths:
                st.warning("Supported raster images nahi mili.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()

            replacements = {}
            fixed = 0
            unchanged = 0
            failed = 0
            skipped = 0

            with zipfile.ZipFile(input_ppt, "r") as ppt_zip:
                total = len(media_paths)

                for i, media_path in enumerate(media_paths, 1):
                    name = Path(media_path).name
                    status.write(f"🖼️ {i}/{total} — {name}")

                    ok, result, output_path = process_media(
                        ppt_zip,
                        media_path,
                        work_dir,
                        mode
                    )

                    if ok:
                        if result == "no_marker":
                            unchanged += 1
                        else:
                            fixed += 1
                            replacements[media_path] = output_path
                    elif result == "unsupported":
                        skipped += 1
                    else:
                        failed += 1
                        st.warning(f"⚠️ {name}: {result}")

                    progress.progress(i / total)
                    gc.collect()

            status.write(
                f"✅ Detection complete — Fixed: {fixed} | "
                f"No marker: {unchanged} | Failed: {failed} | Skipped: {skipped}"
            )

            with st.status("📦 Final PPT ban rahi hai...") as s:
                build_final_ppt(
                    input_ppt,
                    output_ppt,
                    replacements
                )
                s.update(label="✅ Final PPT ready", state="complete")

            st.success(
                f"🎉 Done! {fixed} images me old marking remove karke "
                f"new professional rectangle banaya gaya."
            )

            with open(output_ppt, "rb") as f:
                st.download_button(
                    "⬇️ Download Professional PPT",
                    data=f,
                    file_name=Path(uploaded.name).stem + "_Professional.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )

        except Exception as e:
            st.error("❌ Processing error")
            st.exception(e)
        finally:
            # Keep folder until Streamlit rerun/download lifecycle completes.
            gc.collect()
