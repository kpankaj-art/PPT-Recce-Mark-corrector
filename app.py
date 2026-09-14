import streamlit as st
import os
import io
import gc
import zipfile
import tempfile
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageFile
from google import genai
from google.genai import types


# =========================================================
# SETTINGS
# =========================================================

st.set_page_config(
    page_title="PPT Professional Marking AI",
    page_icon="🖼️",
    layout="wide"
)

# RAM friendly limits
MAX_PPT_SIZE_MB = 300
MAX_AI_SIDE = 1280
JPEG_QUALITY = 82

# Gemini model
GEMINI_MODEL = "gemini-2.5-flash-image"

# Supported raster images
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

# Avoid Pillow loading huge decompression bombs
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    api_key = None

    # Streamlit secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # Environment variable fallback
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Gemini API key nahi mili. "
            "'.streamlit/secrets.toml' me GEMINI_API_KEY add karo."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# AI PROMPT
# =========================================================

AI_PROMPT = """
You are editing a real field-survey / recce photograph.

The photograph may contain a rough black hand-drawn marking,
scribble, circle, irregular line, or freehand outline that is
BAKED INTO THE IMAGE PIXELS.

Your job is to professionally clean that marking.

IMPORTANT:

1. Detect the rough/irregular hand-drawn marking in the image.
2. Remove ONLY the rough marking.
3. Reconstruct the actual photograph underneath the removed marking
   using surrounding pixels, textures, perspective and scene geometry.
4. Identify the SAME real-world object/area that the rough marking
   was intended to highlight.
5. Draw ONE clean professional rectangular outline around that SAME
   target area.
6. The rectangle must have four straight sides.
7. Use a thin professional black outline.
8. Do not fill the rectangle.
9. Do not use a circle, oval, scribble, arrow, glow or decorative effect.
10. Keep the rectangle close to the original rough marking boundaries.
11. Do NOT move the target to another object.
12. Do NOT crop the image.
13. Keep exactly the same aspect ratio and composition.
14. Preserve people, vehicles, buildings, signs, boards, text,
    colors, lighting, shadows, perspective and all unrelated details.
15. Do not add or remove objects.
16. Do not rewrite or invent text.
17. Do not change the overall photograph style.
18. If there is no rough marking, return the image essentially unchanged.
19. Return ONLY the edited image.

The final result should look like a professional site-recce photograph
where an irregular hand-drawn marking has been replaced by a neat,
straight rectangular professional marking.
"""


# =========================================================
# SAVE UPLOADED PPT TO DISK
# =========================================================

def save_uploaded_file(uploaded_file, destination):

    with open(destination, "wb") as out:

        while True:
            chunk = uploaded_file.read(1024 * 1024)

            if not chunk:
                break

            out.write(chunk)


# =========================================================
# FIND MEDIA FILES USED BY SLIDES
# =========================================================

def get_slide_media_references(ppt_path):

    media_paths = set()

    with zipfile.ZipFile(ppt_path, "r") as zin:

        names = set(zin.namelist())

        # Read slide relationship files
        for name in names:

            if not name.startswith("ppt/slides/_rels/"):
                continue

            if not name.endswith(".rels"):
                continue

            try:
                xml_data = zin.read(name)
                root = ET.fromstring(xml_data)

            except Exception:
                continue

            rel_targets = {}

            for rel in root:

                rid = rel.attrib.get("Id")
                target = rel.attrib.get("Target")

                if not rid or not target:
                    continue

                # Example:
                # ../media/image1.jpeg
                if "../media/" in target:

                    filename = target.split("../media/")[-1]

                    rel_targets[rid] = "ppt/media/" + filename

            # Find slide number from relationship path
            # ppt/slides/_rels/slide1.xml.rels

            slide_rel_name = Path(name).name

            if not slide_rel_name.endswith(".rels"):
                continue

            slide_xml_name = slide_rel_name[:-5]

            slide_path = "ppt/slides/" + slide_xml_name

            if slide_path not in names:
                continue

            try:
                slide_xml = zin.read(slide_path)
                slide_root = ET.fromstring(slide_xml)

            except Exception:
                continue

            # Search all elements for r:embed
            for element in slide_root.iter():

                for attr_name, attr_value in element.attrib.items():

                    if attr_name.endswith("}embed"):

                        if attr_value in rel_targets:
                            media_paths.add(
                                rel_targets[attr_value]
                            )

    return sorted(media_paths)


# =========================================================
# PREPARE IMAGE FOR GEMINI
# =========================================================

def prepare_image_for_ai(original_path, ai_path):

    original_width = None
    original_height = None

    with Image.open(original_path) as img:

        original_width, original_height = img.size

        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize only for AI processing
        max_side = max(img.size)

        if max_side > MAX_AI_SIDE:

            scale = MAX_AI_SIDE / max_side

            new_size = (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale))
            )

            img = img.resize(
                new_size,
                Image.Resampling.LANCZOS
            )

        img.save(
            ai_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=False
        )

    gc.collect()

    return original_width, original_height


# =========================================================
# SEND ONE IMAGE TO GEMINI
# =========================================================

def process_image_with_gemini(client, ai_path, output_path):

    with open(ai_path, "rb") as f:
        image_bytes = f.read()

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                AI_PROMPT
            ],

            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )

    finally:

        # Release input image bytes
        del image_bytes
        gc.collect()

    output_bytes = None

    # Find generated image
    if response.parts:

        for part in response.parts:

            if part.inline_data is not None:

                output_bytes = part.inline_data.data
                break

    if not output_bytes:

        raise RuntimeError(
            "Gemini ne image output nahi diya."
        )

    # Save directly to disk
    with open(output_path, "wb") as f:
        f.write(output_bytes)

    del output_bytes
    del response

    gc.collect()


# =========================================================
# RESTORE ORIGINAL DIMENSIONS
# =========================================================

def restore_original_size(
    generated_path,
    final_path,
    original_width,
    original_height
):

    with Image.open(generated_path) as img:

        # Always convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Restore exact original image dimensions
        if img.size != (
            original_width,
            original_height
        ):

            img = img.resize(
                (
                    original_width,
                    original_height
                ),
                Image.Resampling.LANCZOS
            )

        # Save as JPEG
        img.save(
            final_path,
            format="JPEG",
            quality=92,
            optimize=False
        )

    gc.collect()


# =========================================================
# PROCESS ONE MEDIA FILE
# =========================================================

def process_one_media(
    client,
    ppt_zip,
    media_path,
    work_dir
):

    suffix = Path(media_path).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return False, "unsupported"

    # Unique temporary filenames
    safe_name = (
        media_path
        .replace("/", "_")
        .replace("\\", "_")
    )

    original_path = os.path.join(
        work_dir,
        "original_" + safe_name
    )

    ai_path = os.path.join(
        work_dir,
        "ai_" + safe_name + ".jpg"
    )

    generated_path = os.path.join(
        work_dir,
        "generated_" + safe_name + ".png"
    )

    final_path = os.path.join(
        work_dir,
        "final_" + safe_name + ".jpg"
    )

    # Extract ONLY this image
    with ppt_zip.open(media_path, "r") as src:

        with open(original_path, "wb") as dst:

            while True:

                chunk = src.read(1024 * 1024)

                if not chunk:
                    break

                dst.write(chunk)

    # Check image
    try:

        with Image.open(original_path) as test_img:

            test_img.verify()

    except Exception:

        return False, "invalid_image"

    # Prepare AI input
    try:

        width, height = prepare_image_for_ai(
            original_path,
            ai_path
        )

        # Gemini
        process_image_with_gemini(
            client,
            ai_path,
            generated_path
        )

        # Restore exact original dimensions
        restore_original_size(
            generated_path,
            final_path,
            width,
            height
        )

    except Exception as e:

        # Cleanup
        for p in [
            original_path,
            ai_path,
            generated_path,
            final_path
        ]:

            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        gc.collect()

        return False, str(e)

    # Cleanup intermediate files
    for p in [
        original_path,
        ai_path,
        generated_path
    ]:

        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    gc.collect()

    return True, final_path


# =========================================================
# BUILD FINAL PPT
# =========================================================

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

                replacement = replacements.get(filename)

                if replacement and os.path.exists(
                    replacement
                ):

                    # Write replacement image in chunks
                    new_info = zipfile.ZipInfo(
                        filename
                    )

                    new_info.date_time = info.date_time
                    new_info.compress_type = zipfile.ZIP_DEFLATED

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

                    # Copy non-image PPT content
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

    gc.collect()


# =========================================================
# CLEANUP DIRECTORY
# =========================================================

def cleanup_directory(path):

    try:

        shutil.rmtree(
            path,
            ignore_errors=True
        )

    except Exception:
        pass

    gc.collect()


# =========================================================
# UI
# =========================================================

st.title("🖼️ PPT Professional Marking AI")

st.write(
    "PPT ki photos me baked-in rough/hand-drawn markings "
    "ko AI se clean karke professional rectangular marking "
    "banaye."
)

st.info(
    "Designed for large PPT files: ~300 MB and 800+ slides. "
    "Images one-by-one process hoti hain taaki RAM usage low rahe."
)


# =========================================================
# API CHECK
# =========================================================

try:

    client = get_gemini_client()

    st.success("✅ Gemini API connected")

except Exception as e:

    client = None

    st.error(str(e))

    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "PPTX file upload karo",
    type=["pptx"]
)


if uploaded_file:

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.write(
        f"📦 File size: **{file_size_mb:.2f} MB**"
    )

    if file_size_mb > MAX_PPT_SIZE_MB:

        st.error(
            f"File {MAX_PPT_SIZE_MB} MB se badi hai. "
            "Please smaller PPT upload karo."
        )

        st.stop()

    if st.button(
        "🚀 Start Professional Processing",
        type="primary"
    ):

        # Temporary working directory
        work_root = tempfile.mkdtemp(
            prefix="ppt_ai_"
        )

        input_ppt = os.path.join(
            work_root,
            "input.pptx"
        )

        output_ppt = os.path.join(
            work_root,
            "professional_output.pptx"
        )

        try:

            # -------------------------------------------------
            # SAVE INPUT
            # -------------------------------------------------

            with st.status(
                "PPT save ho rahi hai...",
                expanded=True
            ) as status:

                save_uploaded_file(
                    uploaded_file,
                    input_ppt
                )

                status.update(
                    label="PPT successfully save ho gayi",
                    state="complete"
                )

            # -------------------------------------------------
            # FIND IMAGES
            # -------------------------------------------------

            with st.status(
                "PPT ke andar images identify ho rahi hain...",
                expanded=True
            ) as status:

                media_paths = get_slide_media_references(
                    input_ppt
                )

                status.write(
                    f"Found **{len(media_paths)} unique images**"
                )

                status.update(
                    label="Images identify ho gayi",
                    state="complete"
                )

            if not media_paths:

                st.warning(
                    "PPT me supported raster images nahi mili."
                )

                cleanup_directory(
                    work_root
                )

                st.stop()

            # -------------------------------------------------
            # PROCESS IMAGES
            # -------------------------------------------------

            replacements = {}

            success_count = 0
            failed_count = 0
            skipped_count = 0

            progress = st.progress(0)

            status_text = st.empty()

            # Open PPT ZIP only once
            with zipfile.ZipFile(
                input_ppt,
                "r"
            ) as ppt_zip:

                for index, media_path in enumerate(
                    media_paths,
                    start=1
                ):

                    status_text.write(
                        f"Processing image {index}/{len(media_paths)}"
                    )

                    ok, result = process_one_media(
                        client,
                        ppt_zip,
                        media_path,
                        work_root
                    )

                    if ok:

                        replacements[
                            media_path
                        ] = result

                        success_count += 1

                    elif result == "unsupported":

                        skipped_count += 1

                    else:

                        failed_count += 1

                    progress.progress(
                        index / len(media_paths)
                    )

                    # Aggressive cleanup
                    gc.collect()

            status_text.write(
                f"✅ AI processing complete — "
                f"{success_count} processed | "
                f"{failed_count} failed | "
                f"{skipped_count} skipped"
            )

            # -------------------------------------------------
            # BUILD PPT
            # -------------------------------------------------

            with st.status(
                "Final PPT ban rahi hai...",
                expanded=True
            ) as status:

                build_final_ppt(
                    input_ppt,
                    output_ppt,
                    replacements
                )

                status.update(
                    label="Final PPT ready",
                    state="complete"
                )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            output_size_mb = (
                os.path.getsize(output_ppt)
                / (1024 * 1024)
            )

            st.success(
                f"🎉 Processing complete!\n\n"
                f"Images processed: {success_count}\n"
                f"Failed: {failed_count}\n"
                f"Final PPT: {output_size_mb:.2f} MB"
            )

            # Download from disk/file object
            with open(
                output_ppt,
                "rb"
            ) as download_file:

                st.download_button(
                    label="⬇️ Download Professional PPT",
                    data=download_file,
                    file_name=(
                        Path(uploaded_file.name).stem
                        + "_Professional.pptx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.presentationml.presentation"
                    )
                )

            st.caption(
                "Original PPT ki slide layout, positions aur "
                "slide XML ko directly modify nahi kiya gaya. "
                "Sirf referenced raster image files replace ki gayi hain."
            )

        except Exception as e:

            st.error(
                "❌ Processing error:"
            )

            st.exception(e)

        finally:

            # IMPORTANT:
            # Download button ko render hone ke baad Streamlit
            # rerun karega. Isliye immediately cleanup nahi karna.
            #
            # Temporary folder current session ke liye rehne diya
            # gaya hai.

            gc.collect()
