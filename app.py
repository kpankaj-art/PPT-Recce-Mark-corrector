
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
    page_title="PPT Recce Mark Corrector V5",
    page_icon="🟩",
    layout="wide"
)

MAX_PPT_MB = 300
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Final professional marking color = GREEN
GREEN = (0, 200, 0)


def clean_mem():
    gc.collect()


# ============================================================
# GREEN MARKER DETECTION
# ============================================================

def detect_green_markers(bgr):
    """
    V6 green detection.

    The previous version treated natural green storefront/cloth
    areas as marker boxes. V6 first isolates thin green structures
    with a top-hat operation and then applies strict marker geometry.

    It deliberately prefers NO DETECTION over a false green box.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(
        hsv,
        np.array([38, 85, 55], dtype=np.uint8),
        np.array([88, 255, 255], dtype=np.uint8)
    )

    top_hat = cv2.morphologyEx(
        green,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (31, 31)
        )
    )

    top_hat = cv2.morphologyEx(
        top_hat,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),
        iterations=1
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        top_hat,
        8
    )

    H, W = top_hat.shape
    image_area = H * W
    boxes = []

    for i in range(1, n):

        x, y, w, h, area = map(
            int,
            stats[i]
        )

        # Marker must be large enough to be meaningful in a
        # 720p-1024p recce photo.
        if area < max(
            5000,
            int(image_area * 0.003)
        ):
            continue

        if w < 80 or h < 70:
            continue

        if w > W * 0.90 or h > H * 0.96:
            continue

        fill = area / max(
            1,
            w * h
        )

        # Filled green objects are not marker outlines.
        if fill > 0.36:
            continue

        # Edge-connected large green regions are usually cloth,
        # plants, awnings, etc.
        if x <= 1:
            continue

        if y <= 1:
            continue

        if x + w >= W - 1:
            continue

        if y + h >= H - 1:
            continue

        component = np.where(
            labels == i,
            255,
            0
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        vertices = len(approx)

        if vertices < 4 or vertices > 12:
            continue

        aspect = w / max(
            1,
            h
        )

        band = max(
            2,
            int(min(w, h) * 0.035)
        )

        crop = (
            component[y:y+h, x:x+w]
        )

        top_cov = (
            crop[:band, :].mean() / 255.0
        )

        bottom_cov = (
            crop[-band:, :].mean() / 255.0
        )

        left_cov = (
            crop[:, :band].mean() / 255.0
        )

        right_cov = (
            crop[:, -band:].mean() / 255.0
        )

        sides = [
            top_cov,
            bottom_cov,
            left_cov,
            right_cov
        ]

        strong_sides = sum(
            s >= 0.12
            for s in sides
        )

        # Landscape marker: require at least 3 supported sides.
        if aspect >= 0.5:

            if strong_sides < 3:
                continue

        # Tall marker:
        # vertical recce markings can have broken top/bottom
        # strokes, so accept two opposite/strong side structures
        # only when the component is very large and thin.
        else:

            opposite_vertical = (
                left_cov >= 0.075
                and
                right_cov >= 0.075
            )

            large_vertical = (
                area >= max(
                    10000,
                    int(image_area * 0.012)
                )
                and
                h >= W * 0.45
            )

            if not (
                (strong_sides >= 3)
                or
                (
                    opposite_vertical
                    and
                    large_vertical
                )
            ):
                continue

        boxes.append(
            (
                int(x),
                int(y),
                int(w),
                int(h)
            )
        )

    return top_hat, boxes


# ============================================================
# BLACK MARKER DETECTION
# ============================================================

def _max_run(arr):
    best = 0
    current = 0

    for value in arr:
        if value:
            current += 1
            if current > best:
                best = current
        else:
            current = 0

    return best


def _clusters(values, gap=4):
    if not values:
        return []

    groups = []
    current = [values[0]]

    for value in values[1:]:

        if value - current[-1] <= gap:
            current.append(value)
        else:
            groups.append(current)
            current = [value]

    groups.append(current)

    return [
        int(np.mean(group))
        for group in groups
    ]


def find_black_rectangle(bgr):
    """
    Finds the rough black rectangular marking.

    V5 change:
    Instead of asking "are there black pixels?",
    we look for FOUR sides that form a rectangle:
      top + bottom + left + right.

    This is much safer on storefront photos.
    """

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    H, W = gray.shape

    # Pure/dark black. The rough marker is much darker than
    # most brown/grey storefront structures.
    dark = (
        gray < 50
    ).astype(np.uint8)

    y_start = int(H * 0.03)
    y_end = int(H * 0.72)

    x_start = int(W * 0.03)
    x_end = int(W * 0.97)

    row_values = []

    for y in range(
        y_start,
        y_end
    ):

        run = _max_run(
            dark[y, x_start:x_end]
        )

        if run >= max(
            25,
            int(W * 0.07)
        ):

            row_values.append(y)

    col_values = []

    for x in range(
        x_start,
        x_end
    ):

        run = _max_run(
            dark[y_start:y_end, x]
        )

        if run >= max(
            25,
            int(H * 0.07)
        ):

            col_values.append(x)

    rows = _clusters(
        row_values,
        gap=4
    )

    cols = _clusters(
        col_values,
        gap=4
    )

    if len(rows) < 2 or len(cols) < 2:
        return None, None

    best = None

    # Try possible rectangle boundaries.
    for top in rows:

        for bottom in rows:

            if bottom <= top:
                continue

            rect_h = bottom - top

            if rect_h < H * 0.08:
                continue

            if rect_h > H * 0.55:
                continue

            for left in cols:

                for right in cols:

                    if right <= left:
                        continue

                    rect_w = right - left

                    if rect_w < W * 0.12:
                        continue

                    if rect_w > W * 0.90:
                        continue

                    band = max(
                        2,
                        int(min(W, H) * 0.012)
                    )

                    x1 = max(
                        0,
                        left - band
                    )

                    x2 = min(
                        W,
                        right + band + 1
                    )

                    y1 = max(
                        0,
                        top - band
                    )

                    y2 = min(
                        H,
                        bottom + band + 1
                    )

                    top_cov = dark[
                        y1:y2,
                        x1:x2
                    ]

                    # Side-specific support.
                    top_support = dark[
                        y1:y2,
                        left:right + 1
                    ].mean()

                    bottom_support = dark[
                        max(0, bottom - band):
                        min(H, bottom + band + 1),
                        left:right + 1
                    ].mean()

                    left_support = dark[
                        top:bottom + 1,
                        x1:x2
                    ].mean()

                    right_support = dark[
                        top:bottom + 1,
                        max(0, right - band):
                        min(W, right + band + 1)
                    ].mean()

                    supports = [
                        top_support,
                        bottom_support,
                        left_support,
                        right_support
                    ]

                    if min(supports) < 0.035:
                        continue

                    avg_support = float(
                        np.mean(supports)
                    )

                    # A genuine rough rectangle has four
                    # reasonably strong sides.
                    strong = sum(
                        s >= 0.07
                        for s in supports
                    )

                    if strong < 3:
                        continue

                    # Prefer larger, stronger rectangles.
                    size_score = (
                        (rect_w / W) *
                        (rect_h / H)
                    )

                    score = (
                        avg_support * 0.75
                        +
                        size_score * 0.25
                    )

                    if (
                        best is None
                        or score > best[0]
                    ):

                        best = (
                            score,
                            (
                                int(left),
                                int(top),
                                int(right),
                                int(bottom)
                            ),
                            supports
                        )

    if best is None:
        return None, None

    return (
        best[1],
        dark
    )


def black_marker_mask(
    bgr,
    rectangle
):
    """
    V6 black marker cleanup.

    A normal dark storefront frame/sign is rejected.
    The old rough marker is accepted only when its dark component
    has the irregular geometry expected from a hand-drawn outline.
    """

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    H, W = gray.shape

    dark = (
        gray < 55
    ).astype(np.uint8)

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            dark,
            8
        )
    )

    rx1, ry1, rx2, ry2 = rectangle

    pad_x = max(
        15,
        int((rx2 - rx1) * 0.08)
    )

    pad_y = max(
        15,
        int((ry2 - ry1) * 0.08)
    )

    ex1 = max(
        0,
        rx1 - pad_x
    )

    ey1 = max(
        0,
        ry1 - pad_y
    )

    ex2 = min(
        W - 1,
        rx2 + pad_x
    )

    ey2 = min(
        H - 1,
        ry2 + pad_y
    )

    mask = np.zeros_like(
        dark,
        dtype=np.uint8
    )

    main_found = False

    for i in range(1, n):

        x, y, w, h, area = map(
            int,
            stats[i]
        )

        if area < 1500:
            continue

        fill = area / max(
            1,
            w * h
        )

        if fill > 0.20:
            continue

        component = np.where(
            labels == i,
            255,
            0
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        vertices = len(approx)

        rect = cv2.minAreaRect(
            contour
        )

        rw, rh = rect[1]

        rect_area = max(
            1.0,
            rw * rh
        )

        contour_area = max(
            1.0,
            cv2.contourArea(contour)
        )

        shape_ratio = (
            contour_area /
            rect_area
        )

        # Rough marker:
        # irregular enough (5+ vertices) and occupies a
        # meaningful portion of its minimum rectangle.
        rough_shape = (
            5 <= vertices <= 12
            and
            shape_ratio >= 0.25
        )

        intersects = not (
            x + w < ex1
            or x > ex2
            or y + h < ey1
            or y > ey2
        )

        if rough_shape and intersects:

            mask = cv2.bitwise_or(
                mask,
                component
            )

            main_found = True

    if not main_found:
        return (
            np.zeros_like(dark),
            False
        )

    # Cover the actual black stroke only.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        ),
        iterations=1
    )

    return (
        mask,
        True
    )


# ============================================================
# COMBINED DETECTION
# ============================================================

def detect_markings(
    bgr,
    mode
):

    green_pixels, green_boxes = (
        detect_green_markers(bgr)
    )

    # If user explicitly selects Green,
    # never run black detection.
    if mode == "Green":

        if not green_boxes:
            return (
                np.zeros(
                    bgr.shape[:2],
                    dtype=np.uint8
                ),
                [],
                False
            )

        mask = np.zeros(
            bgr.shape[:2],
            dtype=np.uint8
        )

        for x, y, w, h in green_boxes:

            component = cv2.inRange(
                green_pixels[
                    y:y+h,
                    x:x+w
                ],
                1,
                255
            )

            full = np.zeros_like(
                mask
            )

            full[
                y:y+h,
                x:x+w
            ] = component

            mask = cv2.bitwise_or(
                mask,
                full
            )

        return (
            mask,
            green_boxes,
            True
        )

    # For AUTO:
    # If a confident green marker exists, use ONLY those
    # green markers. Do not run black detector on the same
    # image because dark photo structures can create false
    # positives.
    if mode == "Auto" and green_boxes:

        mask = np.zeros(
            bgr.shape[:2],
            dtype=np.uint8
        )

        for x, y, w, h in green_boxes:

            comp = green_pixels[
                y:y+h,
                x:x+w
            ]

            full = np.zeros_like(
                mask
            )

            full[
                y:y+h,
                x:x+w
            ] = comp

            mask = cv2.bitwise_or(
                mask,
                full
            )

        return (
            mask,
            green_boxes,
            True
        )

    # Black mode or AUTO with no green marker.
    rectangle, _ = find_black_rectangle(
        bgr
    )

    if rectangle is None:
        return (
            np.zeros(
                bgr.shape[:2],
                dtype=np.uint8
            ),
            [],
            False
        )

    black_mask, found = (
        black_marker_mask(
            bgr,
            rectangle
        )
    )

    if not found:
        return (
            np.zeros(
                bgr.shape[:2],
                dtype=np.uint8
            ),
            [],
            False
        )

    x1, y1, x2, y2 = rectangle

    return (
        black_mask,
        [
            (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )
        ],
        True
    )


# ============================================================
# REPAIR
# ============================================================

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

    mask, boxes, found = (
        detect_markings(
            bgr,
            mode
        )
    )

    if not found:
        # IMPORTANT:
        # No confident marking = untouched image.
        result = pil_image.convert(
            "RGB"
        )

        del rgb, bgr, mask
        clean_mem()

        return (
            result,
            False
        )

    # Very small radius reduces color bleeding/blue artifacts.
    repaired = cv2.inpaint(
        bgr,
        mask,
        3,
        cv2.INPAINT_TELEA
    )

    H, W = repaired.shape[:2]

    for x, y, w, h in boxes:

        px = max(
            2,
            int(w * 0.006)
        )

        py = max(
            2,
            int(h * 0.006)
        )

        x1 = max(
            0,
            x - px
        )

        y1 = max(
            0,
            y - py
        )

        x2 = min(
            W - 1,
            x + w + px
        )

        y2 = min(
            H - 1,
            y + h + py
        )

        # FINAL PROFESSIONAL MARKING = GREEN
        cv2.rectangle(
            repaired,
            (x1, y1),
            (x2, y2),
            GREEN,
            thickness=max(
                2,
                thickness
            ),
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

    return (
        result,
        True
    )


# ============================================================
# PPT MEDIA REFERENCES
# ============================================================

def get_slide_media_references(
    ppt_path
):

    media = set()

    with zipfile.ZipFile(
        ppt_path,
        "r"
    ) as z:

        names = set(
            z.namelist()
        )

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
                    z.read(
                        rels_name
                    )
                )

            except Exception:
                continue

            relmap = {}

            for rel in root:

                rid = rel.attrib.get(
                    "Id"
                )

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
                    z.read(
                        slide_xml
                    )
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

    return sorted(
        media
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

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

        return (
            False,
            "unsupported",
            None
        )

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

                    dst.write(
                        chunk
                    )

        with Image.open(
            original_path
        ) as img:

            img.load()

            original = img.convert(
                "RGB"
            )

        result, changed = (
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

            elif ext in {
                ".jpg",
                ".jpeg"
            }:

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

            status = "fixed"

        else:

            # Exact original bytes.
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


# ============================================================
# BUILD FINAL PPT
# ============================================================

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

                    new_info = (
                        zipfile.ZipInfo(
                            info.filename
                        )
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
                                    1024 * 1024
                                )

                                if not chunk:
                                    break

                                dst.write(
                                    chunk
                                )

    clean_mem()


# ============================================================
# UI
# ============================================================

st.title(
    "🟩 PPT Recce Mark Corrector V5"
)

st.write(
    "Existing rough marking ko clean green professional "
    "rectangle me convert karta hai."
)

st.info(
    "IMPORTANT: Confident marking na mile to image ko "
    "bilkul untouched rakha jata hai."
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
            prefix="ppt_recce_v5_"
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

                        f.write(
                            chunk
                        )

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
                    "Supported raster images nahi mili."
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
                f"Untouched: {unchanged} | "
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
                f"{fixed} images repair hui aur "
                f"{unchanged} images untouched rahi."
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
