
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

st.set_page_config(page_title="PPT Recce Mark Corrector V4", page_icon="🟩", layout="wide")

MAX_PPT_MB = 300
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Final professional marking color: green
GREEN = (0, 200, 0)


def clean_mem():
    gc.collect()


# ------------------------------------------------------------
# GREEN MARKER
# ------------------------------------------------------------

def detect_green_markers(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Conservative green range. This avoids most cyan/blue photo content.
    mask = cv2.inRange(
        hsv,
        np.array([38, 90, 60], dtype=np.uint8),
        np.array([88, 255, 255], dtype=np.uint8)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    area_total = H * W
    candidates = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area < max(80, area_total * 0.00005):
            continue
        if w < W * 0.08 and h < H * 0.08:
            continue
        if w > W * 0.96 and h > H * 0.96:
            continue

        candidates.append((area, i, x, y, w, h))

    if not candidates:
        return np.zeros_like(mask), []

    # If several green objects exist, keep meaningful marker-sized ones.
    largest = max(x[0] for x in candidates)
    selected = [
        x for x in candidates
        if x[0] >= max(80, largest * 0.10)
    ]

    result = np.zeros_like(mask)
    boxes = []

    for area, i, x, y, w, h in selected:
        component = np.where(labels == i, 255, 0).astype(np.uint8)

        # Remove anti-aliased green edge, but keep mask close to stroke.
        component = cv2.dilate(
            component,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        result = cv2.bitwise_or(result, component)
        boxes.append((int(x), int(y), int(w), int(h)))

    return result, boxes


# ------------------------------------------------------------
# BLACK ROUGH MARKER
# ------------------------------------------------------------

def detect_black_markers(bgr):
    """
    Conservative black-marker detector.

    Key idea:
    Normal photo text/objects contain many dark pixels.
    A hand-drawn marker has a comparatively THICK continuous stroke.

    We therefore:
      1. find very dark pixels,
      2. keep only thick stroke cores using distance transform,
      3. connect the stroke,
      4. accept only large outline-like components away from the GPS band.

    If confidence is low, return NO DETECTION.
    That is intentional: an unmarked image must remain untouched.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    total = H * W

    dark = (gray < 70).astype(np.uint8)

    dist = cv2.distanceTransform(
        dark,
        cv2.DIST_L2,
        5
    )

    # Marker stroke is substantially thicker than most text/edges.
    thick = ((dark > 0) & (dist >= 4.2)).astype(np.uint8) * 255

    thick = cv2.morphologyEx(
        thick,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        thick,
        8
    )

    candidates = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        box_area = max(1, w * h)
        fill = area / box_area

        # Exclude the usual bottom GPS/location overlay.
        if y > H * 0.74:
            continue
        if y + h > H * 0.78:
            continue

        # Avoid lines touching the photo edge.
        if x <= 2 or y <= 2 or x + w >= W - 2:
            continue

        if area < max(400, total * 0.00025):
            continue

        # The marker in the sample is a large rough rectangle.
        # Accept either a wide horizontal or tall vertical outline.
        horizontal = (
            w > W * 0.40 and
            h > H * 0.075
        )

        vertical = (
            h > H * 0.35 and
            w > W * 0.06
        )

        if not (horizontal or vertical):
            continue

        # Natural solid dark objects generally have higher fill.
        if fill > 0.10:
            continue

        score = (
            (w / W) *
            (h / H) *
            (1.0 - fill)
        )

        candidates.append(
            (score, i, x, y, w, h, area, fill)
        )

    if not candidates:
        return np.zeros_like(gray), []

    candidates.sort(reverse=True)

    best_score = candidates[0][0]

    # Multiple components can belong to multiple marker boxes.
    keep = [
        c for c in candidates
        if c[0] >= best_score * 0.45
    ]

    result = np.zeros_like(gray)
    boxes = []

    for score, i, x, y, w, h, area, fill in keep:

        component = np.where(
            labels == i,
            255,
            0
        ).astype(np.uint8)

        # IMPORTANT:
        # The previous version left the old black line behind.
        # Expand enough to cover the complete thick marker stroke.
        component = cv2.dilate(
            component,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (15, 15)
            ),
            iterations=1
        )

        result = cv2.bitwise_or(
            result,
            component
        )

        boxes.append(
            (int(x), int(y), int(w), int(h))
        )

    return result, boxes


# ------------------------------------------------------------
# ALL MARKERS
# ------------------------------------------------------------

def detect_all_markers(bgr, mode):
    green_mask, green_boxes = detect_green_markers(bgr)
    black_mask, black_boxes = detect_black_markers(bgr)

    if mode == "Green":
        return green_mask, green_boxes, "green"

    if mode == "Black":
        return black_mask, black_boxes, "black"

    # AUTO:
    # Process only HIGH-CONFIDENCE marker candidates.
    # If neither is confidently detected, do absolutely nothing.
    combined = cv2.bitwise_or(
        green_mask,
        black_mask
    )

    boxes = []
    boxes.extend(green_boxes)
    boxes.extend(black_boxes)

    if not boxes or np.count_nonzero(combined) < 50:
        return np.zeros_like(green_mask), [], None

    return combined, boxes, "mixed"


# ------------------------------------------------------------
# REPAIR IMAGE
# ------------------------------------------------------------

def repair_image(
    pil_image,
    mode="Auto",
    thickness=4
):
    rgb = np.asarray(
        pil_image.convert("RGB")
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    mask, boxes, marker_type = detect_all_markers(
        bgr,
        mode
    )

    if not boxes or np.count_nonzero(mask) < 50:
        return (
            pil_image.convert("RGB"),
            False,
            None
        )

    # Small inpainting radius prevents the blue/blur artifacts
    # seen in the previous green-marker version.
    radius = 3

    repaired = cv2.inpaint(
        bgr,
        mask,
        radius,
        cv2.INPAINT_TELEA
    )

    H, W = repaired.shape[:2]

    # Draw ONE clean green rectangle for each detected marker.
    for x, y, w, h in boxes:

        # Tiny expansion around original marker bounds.
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
            GREEN,
            thickness=max(2, thickness),
            lineType=cv2.LINE_AA
        )

    result = Image.fromarray(
        cv2.cvtColor(
            repaired,
            cv2.COLOR_BGR2RGB
        )
    )

    del rgb, bgr, mask, repaired
    clean_mem()

    return result, True, marker_type


# ------------------------------------------------------------
# PPT IMAGE DISCOVERY
# ------------------------------------------------------------

def get_slide_media_references(ppt_path):
    media = set()

    with zipfile.ZipFile(
        ppt_path,
        "r"
    ) as z:

        names = set(z.namelist())

        for rels_name in names:

            if not rels_name.startswith(
                "ppt/slides/_rels/"
            ):
                continue

            if not rels_name.endswith(
                ".rels"
            ):
                continue

            try:
                root = ET.fromstring(
                    z.read(rels_name)
                )
            except Exception:
                continue

            relmap = {}

            for rel in root:

                rid = rel.attrib.get("Id")
                target = rel.attrib.get(
                    "Target",
                    ""
                )

                if (
                    rid
                    and
                    "../media/" in target
                ):

                    relmap[rid] = (
                        "ppt/media/"
                        +
                        target.split(
                            "../media/"
                        )[-1]
                    )

            slide_name = Path(
                rels_name
            ).name[:-5]

            slide_xml = (
                "ppt/slides/"
                +
                slide_name
            )

            if slide_xml not in names:
                continue

            try:
                root = ET.fromstring(
                    z.read(slide_xml)
                )
            except Exception:
                continue

            for element in root.iter():

                for attr, value in (
                    element.attrib.items()
                ):

                    if (
                        attr.endswith(
                            "}embed"
                        )
                        and
                        value in relmap
                    ):

                        media.add(
                            relmap[value]
                        )

    return sorted(media)


# ------------------------------------------------------------
# PROCESS ONE IMAGE
# ------------------------------------------------------------

def process_media(
    ppt_zip,
    media_path,
    work_dir,
    mode,
    thickness
):

    ext = Path(
        media_path
    ).suffix.lower()

    if ext not in SUPPORTED:
        return False, "unsupported", None

    safe = (
        media_path
        .replace("/", "_")
        .replace("\\", "_")
    )

    original_path = os.path.join(
        work_dir,
        "original_" + safe
    )

    fixed_path = os.path.join(
        work_dir,
        "fixed_" + safe
    )

    try:

        with ppt_zip.open(
            media_path,
            "r"
        ) as src:

            with open(
                original_path,
                "wb"
            ) as dst:

                while True:

                    chunk = src.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    dst.write(chunk)

        with Image.open(
            original_path
        ) as img:

            img.load()

            original = img.convert(
                "RGB"
            )

        result, changed, marker_type = (
            repair_image(
                original,
                mode,
                thickness
            )
        )

        if changed:

            if ext == ".png":

                result.save(
                    fixed_path,
                    format="PNG",
                    optimize=False
                )

            elif ext in {".jpg", ".jpeg"}:

                result.save(
                    fixed_path,
                    format="JPEG",
                    quality=94,
                    optimize=False
                )

            elif ext == ".webp":

                result.save(
                    fixed_path,
                    format="WEBP",
                    quality=94
                )

            elif ext == ".bmp":

                result.save(
                    fixed_path,
                    format="BMP"
                )

            result.close()

            status = (
                "fixed_" +
                str(marker_type)
            )

        else:

            # CRITICAL:
            # No confident marking = EXACT original bytes.
            shutil.copyfile(
                original_path,
                fixed_path
            )

            status = "unchanged"

        original.close()

        try:
            os.remove(
                original_path
            )
        except Exception:
            pass

        clean_mem()

        return (
            True,
            status,
            fixed_path
        )

    except Exception as e:

        for p in [
            original_path,
            fixed_path
        ]:

            try:

                if os.path.exists(p):
                    os.remove(p)

            except Exception:
                pass

        clean_mem()

        return (
            False,
            str(e),
            None
        )


# ------------------------------------------------------------
# BUILD PPT
# ------------------------------------------------------------

def build_final_ppt(
    input_ppt,
    output_ppt,
    replacements
):

    with zipfile.ZipFile(
        input_ppt,
        "r"
    ) as zin:

        with zipfile.ZipFile(
            output_ppt,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as zout:

            for info in zin.infolist():

                replacement = (
                    replacements.get(
                        info.filename
                    )
                )

                if (
                    replacement
                    and
                    os.path.exists(
                        replacement
                    )
                ):

                    new_info = zipfile.ZipInfo(
                        info.filename
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
                                    1024 * 1024
                                )

                                if not chunk:
                                    break

                                dst.write(chunk)

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
                                    1024 * 1024
                                )

                                if not chunk:
                                    break

                                dst.write(chunk)

    clean_mem()


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.title(
    "🟩 PPT Recce Mark Corrector V4"
)

st.write(
    "Sirf existing rough marking wali images ko repair karta hai. "
    "Unmarked images ko touch nahi karta."
)

st.info(
    "Final marking hamesha GREEN professional rectangle hogi."
)

mode = st.selectbox(
    "Marking type",
    [
        "Auto",
        "Green",
        "Black"
    ]
)

thickness = st.slider(
    "Green rectangle thickness",
    2,
    8,
    4
)

uploaded = st.file_uploader(
    "PPTX upload karo — maximum 300 MB",
    type=["pptx"]
)

if uploaded:

    size_mb = (
        uploaded.size /
        (1024 * 1024)
    )

    st.write(
        f"📦 PPT Size: **{size_mb:.2f} MB**"
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
            prefix="ppt_recce_v4_"
        )

        input_ppt = os.path.join(
            work_dir,
            "input.pptx"
        )

        output_ppt = os.path.join(
            work_dir,
            "Professional.pptx"
        )

        try:

            with st.status(
                "📥 PPT save ho rahi hai..."
            ) as s:

                with open(
                    input_ppt,
                    "wb"
                ) as f:

                    while True:

                        chunk = uploaded.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        f.write(chunk)

                s.update(
                    label="✅ PPT save ho gayi",
                    state="complete"
                )

            with st.status(
                "🔎 Images identify ho rahi hain..."
            ) as s:

                media_paths = (
                    get_slide_media_references(
                        input_ppt
                    )
                )

                s.write(
                    f"Unique images: **{len(media_paths)}**"
                )

                s.update(
                    label="✅ Images identify ho gayi",
                    state="complete"
                )

            if not media_paths:

                st.warning(
                    "Supported images nahi mili."
                )

                st.stop()

            progress = st.progress(
                0
            )

            status = st.empty()

            replacements = {}

            fixed = 0
            unchanged = 0
            failed = 0
            skipped = 0

            with zipfile.ZipFile(
                input_ppt,
                "r"
            ) as ppt_zip:

                total = len(
                    media_paths
                )

                for i, media_path in enumerate(
                    media_paths,
                    1
                ):

                    name = Path(
                        media_path
                    ).name

                    status.write(
                        f"🖼️ {i}/{total} — {name}"
                    )

                    ok, result, fixed_path = (
                        process_media(
                            ppt_zip,
                            media_path,
                            work_dir,
                            mode,
                            thickness
                        )
                    )

                    if ok:

                        if result == "unchanged":

                            unchanged += 1

                        else:

                            fixed += 1

                            replacements[
                                media_path
                            ] = fixed_path

                    elif result == "unsupported":

                        skipped += 1

                    else:

                        failed += 1

                        st.warning(
                            f"⚠️ {name}: {result}"
                        )

                    progress.progress(
                        i / total
                    )

                    clean_mem()

            status.write(
                f"✅ Complete — "
                f"Fixed: {fixed} | "
                f"Unchanged: {unchanged} | "
                f"Failed: {failed} | "
                f"Skipped: {skipped}"
            )

            with st.status(
                "📦 Final PPT ban rahi hai..."
            ) as s:

                build_final_ppt(
                    input_ppt,
                    output_ppt,
                    replacements
                )

                s.update(
                    label="✅ Final PPT ready",
                    state="complete"
                )

            st.success(
                f"🎉 Done! "
                f"{fixed} marked images repair hui. "
                f"{unchanged} unmarked images ko untouched rakha gaya."
            )

            with open(
                output_ppt,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ Download Professional PPT",
                    data=f,
                    file_name=(
                        Path(
                            uploaded.name
                        ).stem
                        +
                        "_Professional.pptx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "presentationml.presentation"
                    ),
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "❌ Processing error"
            )

            st.exception(e)
