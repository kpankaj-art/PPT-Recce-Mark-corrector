
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

st.set_page_config(page_title="PPT Recce Mark Corrector V3", page_icon="🔲", layout="wide")

MAX_PPT_MB = 300
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def clean_mem():
    gc.collect()


def green_marker_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Survey-marker green: deliberately conservative to avoid
    # changing blue/grey/photo areas.
    raw = cv2.inRange(
        hsv,
        np.array([38, 85, 65], dtype=np.uint8),
        np.array([88, 255, 255], dtype=np.uint8)
    )

    # Join small breaks in the hand-drawn stroke.
    k = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if n <= 1:
        return np.zeros_like(raw), None

    H, W = raw.shape
    image_area = H * W
    best = None

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < max(40, image_area * 0.00002):
            continue

        # A real marker is normally a sizeable connected stroke.
        if w < W * 0.08 and h < H * 0.08:
            continue
        if w > W * 0.95 and h > H * 0.95:
            continue

        score = area
        # Prefer elongated outline-like components.
        if w > W * 0.15 or h > H * 0.15:
            score *= 1.25

        if best is None or score > best[0]:
            best = (score, i, x, y, w, h)

    if best is None:
        return np.zeros_like(raw), None

    _, i, x, y, w, h = best
    mask = np.where(labels == i, 255, 0).astype(np.uint8)

    # Remove anti-aliased green edges too, but keep the repair narrow.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    return mask, (x, y, w, h)


def black_marker_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Dark pixels only. We then select ONE outline-like connected
    # component instead of treating every black pixel in the photo
    # as the marker.
    dark = cv2.inRange(gray, 0, 65)

    # Join broken sections of a thick hand-drawn line.
    join_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    connected = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, join_kernel, iterations=2
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected, 8
    )

    H, W = gray.shape
    image_area = H * W
    candidates = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        box_area = max(1, w * h)
        fill = area / box_area

        # Rough rectangle has a sizeable bounding box but does NOT
        # fill the whole box.
        if w < W * 0.15:
            continue
        if h < H * 0.05:
            continue
        if w > W * 0.92:
            continue
        if h > H * 0.75:
            continue
        if area < max(700, image_area * 0.0003):
            continue
        if fill > 0.30:
            continue

        # Score large, low-fill, outline-like components.
        score = (
            (w / W) *
            (h / H) *
            (1.0 - fill)
        )

        candidates.append(
            (score, i, x, y, w, h, area, fill)
        )

    if not candidates:
        return np.zeros_like(gray), None

    candidates.sort(reverse=True)
    _, i, x, y, w, h, area, fill = candidates[0]

    # CRITICAL FIX:
    # use ONLY the selected connected marker component.
    # Do NOT erase the bounding-box interior.
    mask = np.where(labels == i, 255, 0).astype(np.uint8)

    # Small expansion catches anti-aliased edges of the old marker.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        ),
        iterations=1
    )

    return mask, (x, y, w, h)


def detect_marker(bgr, mode):
    if mode == "Green":
        return green_marker_mask(bgr), "green"

    if mode == "Black":
        return black_marker_mask(bgr), "black"

    green_mask, green_box = green_marker_mask(bgr)
    green_area = int(np.count_nonzero(green_mask))

    if green_box is not None and green_area >= 80:
        return (green_mask, green_box), "green"

    black_mask, black_box = black_marker_mask(bgr)
    black_area = int(np.count_nonzero(black_mask))

    if black_box is not None and black_area >= 100:
        return (black_mask, black_box), "black"

    return (np.zeros(bgr.shape[:2], dtype=np.uint8), None), None


def repair_image(pil_image, mode="Auto", rectangle_thickness=3):
    """
    Remove only the detected marker stroke with OpenCV inpainting,
    then draw a clean rectangle using the marker's bounding box.
    """
    rgb = np.asarray(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    (mask, bbox), marker_type = detect_marker(bgr, mode)

    if bbox is None or np.count_nonzero(mask) < 30:
        return pil_image.convert("RGB"), False, None, None

    # Inpaint radius intentionally small. A large radius was causing
    # the blue/blurred artifacts seen in the earlier version.
    radius = 3 if marker_type == "green" else 4

    repaired = cv2.inpaint(
        bgr,
        mask,
        radius,
        cv2.INPAINT_TELEA
    )

    x, y, w, h = bbox
    H, W = repaired.shape[:2]

    # A tiny expansion makes the professional rectangle sit around
    # the same target without adding a second oversized rectangle.
    px = max(2, int(w * 0.006))
    py = max(2, int(h * 0.006))

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(W - 1, x + w + px)
    y2 = min(H - 1, y + h + py)

    cv2.rectangle(
        repaired,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        thickness=max(2, rectangle_thickness),
        lineType=cv2.LINE_AA
    )

    result = Image.fromarray(
        cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
    )

    del rgb, bgr, mask, repaired
    clean_mem()

    return result, True, marker_type, (x1, y1, x2, y2)


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
                    relmap[rid] = "ppt/media/" + target.split("../media/")[-1]

            slide_name = Path(rels_name).name[:-5]
            slide_xml = "ppt/slides/" + slide_name

            if slide_xml not in names:
                continue

            try:
                root = ET.fromstring(z.read(slide_xml))
            except Exception:
                continue

            for element in root.iter():
                for attr, value in element.attrib.items():
                    if attr.endswith("}embed") and value in relmap:
                        media.add(relmap[value])

    return sorted(media)


def save_image_same_format(img, path, ext):
    ext = ext.lower()

    if ext == ".png":
        img.save(path, format="PNG", optimize=False)
    elif ext in {".jpg", ".jpeg"}:
        img.save(path, format="JPEG", quality=94, optimize=False)
    elif ext == ".webp":
        img.save(path, format="WEBP", quality=94)
    elif ext == ".bmp":
        img.save(path, format="BMP")
    else:
        img.save(path, format="PNG")


def process_media(ppt_zip, media_path, work_dir, mode, thickness):
    ext = Path(media_path).suffix.lower()

    if ext not in SUPPORTED:
        return False, "unsupported", None, None

    safe = media_path.replace("/", "_").replace("\\", "_")
    original_path = os.path.join(work_dir, "original_" + safe)
    fixed_path = os.path.join(work_dir, "fixed_" + safe)

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

        result, changed, marker_type, box = repair_image(
            original, mode, thickness
        )

        if changed:
            save_image_same_format(result, fixed_path, ext)
            result.close()
        else:
            shutil.copyfile(original_path, fixed_path)

        original.close()

        try:
            os.remove(original_path)
        except Exception:
            pass

        clean_mem()
        return True, ("fixed_" + marker_type if changed else "unchanged"), fixed_path, box

    except Exception as e:
        for p in [original_path, fixed_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        clean_mem()
        return False, str(e), None, None


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
                    ni = zipfile.ZipInfo(info.filename)
                    ni.date_time = info.date_time
                    ni.compress_type = zipfile.ZIP_DEFLATED

                    with open(replacement, "rb") as src:
                        with zout.open(ni, "w") as dst:
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

    clean_mem()


# ============================================================
# UI
# ============================================================

st.title("🔲 PPT Recce Mark Corrector V3")

st.write(
    "Baked-in rough marking ko remove karke usi target par "
    "clean professional rectangle banata hai."
)

st.info(
    "V3: marker stroke ko hi mask karta hai. Bounding-box ka "
    "poora interior erase nahi karta."
)

mode = st.selectbox(
    "Marking type",
    ["Auto", "Green", "Black"]
)

thickness = st.slider(
    "Professional rectangle thickness",
    min_value=2,
    max_value=8,
    value=3
)

uploaded = st.file_uploader(
    "PPTX upload karo — maximum 300 MB",
    type=["pptx"]
)

if uploaded:
    size_mb = uploaded.size / (1024 * 1024)
    st.write(f"📦 PPT Size: **{size_mb:.2f} MB**")

    if size_mb > MAX_PPT_MB:
        st.error("❌ PPT 300 MB se badi hai.")
        st.stop()

    if st.button(
        "🚀 Start Processing",
        type="primary",
        use_container_width=True
    ):
        work_dir = tempfile.mkdtemp(prefix="ppt_recc_v3_")
        input_ppt = os.path.join(work_dir, "input.pptx")
        output_ppt = os.path.join(work_dir, "Professional.pptx")

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
                s.write(f"Unique images: **{len(media_paths)}**")
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

                    ok, result, fixed_path, box = process_media(
                        ppt_zip,
                        media_path,
                        work_dir,
                        mode,
                        thickness
                    )

                    if ok:
                        if result == "unchanged":
                            unchanged += 1
                        else:
                            fixed += 1
                            replacements[media_path] = fixed_path
                    elif result == "unsupported":
                        skipped += 1
                    else:
                        failed += 1
                        st.warning(f"⚠️ {name}: {result}")

                    progress.progress(i / total)
                    clean_mem()

            status.write(
                f"✅ Complete — Fixed: {fixed} | "
                f"Unchanged: {unchanged} | Failed: {failed} | Skipped: {skipped}"
            )

            with st.status("📦 Final PPT ban rahi hai...") as s:
                build_final_ppt(input_ppt, output_ppt, replacements)
                s.update(label="✅ Final PPT ready", state="complete")

            st.success(
                f"🎉 Done! {fixed} images me marking repair hui."
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
