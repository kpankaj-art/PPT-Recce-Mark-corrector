import streamlit as st
import os
import gc
import zipfile
import tempfile
import shutil
import base64
import xml.etree.ElementTree as ET

from pathlib import Path

from PIL import Image, ImageFile

from google import genai
from google.genai import types

from openai import OpenAI


# ============================================================
# STREAMLIT SETTINGS
# ============================================================

st.set_page_config(
    page_title="PPT Professional Marking AI",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# APP SETTINGS
# ============================================================

MAX_PPT_SIZE_MB = 300

# Gemini/OpenAI ko bhejne se pehle image ka maximum side
MAX_AI_SIDE = 1280

# Input JPEG quality
JPEG_QUALITY = 82

# Final image quality
FINAL_JPEG_QUALITY = 92

# Supported raster images
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

# Pillow safety
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# MODELS
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash-image"
OPENAI_MODEL = "gpt-image-1"


# ============================================================
# AI PROMPT
# ============================================================

AI_PROMPT = """
You are editing a real field-survey / recce photograph.

The photograph may contain a rough black hand-drawn marking,
scribble, circle, irregular line, or freehand outline that is
BAKED INTO THE IMAGE PIXELS.

Your job is to professionally clean that marking.

VERY IMPORTANT:

1. Detect the rough/irregular hand-drawn marking in the image.

2. Remove ONLY the rough hand-drawn marking.

3. Reconstruct the actual photograph underneath the removed
   marking using surrounding pixels, textures, perspective,
   lighting and scene geometry.

4. Identify the SAME real-world object or area that the rough
   marking was intended to highlight.

5. Draw ONE clean professional rectangular outline around
   that SAME target area.

6. The new rectangle must have four straight sides.

7. Use a thin professional BLACK outline.

8. Do NOT fill the rectangle.

9. Do NOT use a circle.

10. Do NOT use an oval.

11. Do NOT use a scribble.

12. Do NOT use an arrow.

13. Do NOT use glow, shadow or decorative effects.

14. Keep the new rectangle close to the original rough
    marking boundaries.

15. Do NOT move the target to another object.

16. Do NOT crop the image.

17. Keep exactly the same aspect ratio and composition.

18. Preserve people, vehicles, buildings, signs, boards,
    text, colors, lighting, shadows and perspective.

19. Do NOT add new objects.

20. Do NOT remove unrelated objects.

21. Do NOT rewrite or invent text.

22. Do NOT change the overall photograph style.

23. If there is no rough marking, return the image essentially
    unchanged.

24. The final result must look like a professional site-recce
    photograph.

25. The rough hand-drawn marking must be replaced by a neat,
    straight, professional rectangular marking.

26. Return ONLY the edited image.
"""


# ============================================================
# UI HEADER
# ============================================================

st.title("🖼️ PPT Professional Marking AI")

st.write(
    "PPT ki photos me baked-in rough/hand-drawn markings "
    "ko AI se clean karke professional rectangular marking "
    "banaye."
)

st.info(
    "Designed for large PPT files — approximately 300 MB "
    "and 800+ slides. Images one-by-one process hoti hain "
    "taaki RAM usage low rahe."
)


# ============================================================
# AI PROVIDER
# ============================================================

st.subheader("🤖 AI Provider")

provider = st.radio(
    "Kis AI ki API use karni hai?",
    [
        "Google Gemini",
        "OpenAI"
    ],
    horizontal=True
)


# ============================================================
# API KEY INPUT
# ============================================================

if provider == "Google Gemini":

    st.subheader("🔑 Gemini API Key")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Google AI Studio se Gemini API key paste karo."
    )

    if not api_key:
        st.info(
            "Gemini API key enter karo."
        )
        st.stop()

    try:

        client = genai.Client(
            api_key=api_key
        )

        st.success(
            "✅ Gemini API ready"
        )

    except Exception as e:

        st.error(
            f"❌ Gemini API error: {e}"
        )

        st.stop()


else:

    st.subheader("🔑 OpenAI API Key")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="OpenAI Platform se API key paste karo."
    )

    if not api_key:
        st.info(
            "OpenAI API key enter karo."
        )
        st.stop()

    try:

        client = OpenAI(
            api_key=api_key
        )

        st.success(
            "✅ OpenAI API ready"
        )

    except Exception as e:

        st.error(
            f"❌ OpenAI API error: {e}"
        )

        st.stop()


# ============================================================
# SAVE UPLOADED FILE TO DISK
# ============================================================

def save_uploaded_file(uploaded_file, destination):

    with open(destination, "wb") as output:

        while True:

            chunk = uploaded_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            output.write(chunk)


# ============================================================
# FIND ALL IMAGES USED BY PPT SLIDES
# ============================================================

def get_slide_media_references(ppt_path):

    media_paths = set()

    with zipfile.ZipFile(
        ppt_path,
        "r"
    ) as zin:

        names = set(
            zin.namelist()
        )

        # ----------------------------------------------
        # Read slide relationship files
        # ----------------------------------------------

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

                rels_xml = zin.read(
                    rels_name
                )

                rels_root = ET.fromstring(
                    rels_xml
                )

            except Exception:

                continue

            relationship_map = {}

            for rel in rels_root:

                rid = rel.attrib.get(
                    "Id"
                )

                target = rel.attrib.get(
                    "Target"
                )

                if not rid or not target:
                    continue

                if "../media/" in target:

                    filename = target.split(
                        "../media/"
                    )[-1]

                    relationship_map[
                        rid
                    ] = "ppt/media/" + filename

            # ------------------------------------------
            # Get actual slide XML
            # ------------------------------------------

            rel_filename = Path(
                rels_name
            ).name

            if not rel_filename.endswith(
                ".rels"
            ):
                continue

            slide_filename = rel_filename[
                :-5
            ]

            slide_xml_path = (
                "ppt/slides/"
                + slide_filename
            )

            if slide_xml_path not in names:
                continue

            try:

                slide_xml = zin.read(
                    slide_xml_path
                )

                slide_root = ET.fromstring(
                    slide_xml
                )

            except Exception:

                continue

            # ------------------------------------------
            # Find r:embed references
            # ------------------------------------------

            for element in slide_root.iter():

                for attr_name, attr_value in (
                    element.attrib.items()
                ):

                    if attr_name.endswith(
                        "}embed"
                    ):

                        if attr_value in relationship_map:

                            media_paths.add(
                                relationship_map[
                                    attr_value
                                ]
                            )

    return sorted(
        media_paths
    )


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image_for_ai(
    original_path,
    ai_path
):

    with Image.open(
        original_path
    ) as img:

        original_width = img.width
        original_height = img.height

        # ------------------------------------------
        # Convert to RGB
        # ------------------------------------------

        if img.mode != "RGB":

            img = img.convert(
                "RGB"
            )

        # ------------------------------------------
        # Resize for AI
        # ------------------------------------------

        max_side = max(
            img.size
        )

        if max_side > MAX_AI_SIDE:

            scale = (
                MAX_AI_SIDE /
                max_side
            )

            new_width = max(
                1,
                int(img.width * scale)
            )

            new_height = max(
                1,
                int(img.height * scale)
            )

            img = img.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.LANCZOS
            )

        # ------------------------------------------
        # Save temporary AI image
        # ------------------------------------------

        img.save(
            ai_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=False
        )

    gc.collect()

    return (
        original_width,
        original_height
    )


# ============================================================
# GEMINI PROCESSING
# ============================================================

def process_with_gemini(
    client,
    ai_path,
    output_path
):

    with open(
        ai_path,
        "rb"
    ) as f:

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
                response_modalities=[
                    "IMAGE"
                ]
            )
        )

    finally:

        del image_bytes

        gc.collect()

    generated_data = None

    if response.parts:

        for part in response.parts:

            if part.inline_data is not None:

                generated_data = (
                    part.inline_data.data
                )

                break

    if not generated_data:

        raise RuntimeError(
            "Gemini ne image output nahi diya."
        )

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            generated_data
        )

    del generated_data
    del response

    gc.collect()


# ============================================================
# OPENAI PROCESSING
# ============================================================

def process_with_openai(
    client,
    ai_path,
    output_path
):

    # OpenAI image edit
    with open(
        ai_path,
        "rb"
    ) as image_file:

        result = client.images.edit(

            model=OPENAI_MODEL,

            image=image_file,

            prompt=AI_PROMPT,

            size="auto",

            quality="auto",

            input_fidelity="high"
        )

    if not result.data:

        raise RuntimeError(
            "OpenAI ne image output nahi diya."
        )

    image_result = result.data[0]

    if not image_result.b64_json:

        raise RuntimeError(
            "OpenAI response me image data nahi mila."
        )

    image_bytes = base64.b64decode(
        image_result.b64_json
    )

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            image_bytes
        )

    del image_bytes
    del result

    gc.collect()


# ============================================================
# RESTORE ORIGINAL IMAGE SIZE
# ============================================================

def restore_original_size(
    generated_path,
    final_path,
    original_width,
    original_height
):

    with Image.open(
        generated_path
    ) as img:

        if img.mode != "RGB":

            img = img.convert(
                "RGB"
            )

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

        img.save(
            final_path,
            format="JPEG",
            quality=FINAL_JPEG_QUALITY,
            optimize=False
        )

    gc.collect()


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    client,
    provider,
    ppt_zip,
    media_path,
    work_dir
):

    extension = Path(
        media_path
    ).suffix.lower()

    # ------------------------------------------
    # Skip unsupported formats
    # ------------------------------------------

    if extension not in SUPPORTED_EXTENSIONS:

        return (
            False,
            "unsupported"
        )

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

    try:

        # ==========================================
        # EXTRACT ONLY ONE IMAGE
        # ==========================================

        with ppt_zip.open(
            media_path,
            "r"
        ) as source:

            with open(
                original_path,
                "wb"
            ) as destination:

                while True:

                    chunk = source.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    destination.write(
                        chunk
                    )

        # ==========================================
        # CHECK IMAGE
        # ==========================================

        try:

            with Image.open(
                original_path
            ) as test_img:

                test_img.verify()

        except Exception:

            return (
                False,
                "invalid_image"
            )

        # ==========================================
        # PREPARE IMAGE
        # ==========================================

        (
            original_width,
            original_height
        ) = prepare_image_for_ai(
            original_path,
            ai_path
        )

        # ==========================================
        # AI EDIT
        # ==========================================

        if provider == "Google Gemini":

            process_with_gemini(
                client,
                ai_path,
                generated_path
            )

        else:

            process_with_openai(
                client,
                ai_path,
                generated_path
            )

        # ==========================================
        # RESTORE ORIGINAL SIZE
        # ==========================================

        restore_original_size(
            generated_path,
            final_path,
            original_width,
            original_height
        )

        # ==========================================
        # DELETE TEMP FILES
        # ==========================================

        for temp_file in [
            original_path,
            ai_path,
            generated_path
        ]:

            try:

                if os.path.exists(
                    temp_file
                ):

                    os.remove(
                        temp_file
                    )

            except Exception:
                pass

        gc.collect()

        return (
            True,
            final_path
        )

    except Exception as e:

        # Cleanup after error

        for temp_file in [
            original_path,
            ai_path,
            generated_path,
            final_path
        ]:

            try:

                if os.path.exists(
                    temp_file
                ):

                    os.remove(
                        temp_file
                    )

            except Exception:
                pass

        gc.collect()

        return (
            False,
            str(e)
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

                # ======================================
                # REPLACE IMAGE
                # ======================================

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
                    ) as source:

                        with zout.open(
                            new_info,
                            "w"
                        ) as destination:

                            while True:

                                chunk = source.read(
                                    1024 * 1024
                                )

                                if not chunk:
                                    break

                                destination.write(
                                    chunk
                                )

                # ======================================
                # COPY EVERYTHING ELSE
                # ======================================

                else:

                    with zin.open(
                        info,
                        "r"
                    ) as source:

                        with zout.open(
                            info,
                            "w"
                        ) as destination:

                            while True:

                                chunk = source.read(
                                    1024 * 1024
                                )

                                if not chunk:
                                    break

                                destination.write(
                                    chunk
                                )

    gc.collect()


# ============================================================
# CLEANUP
# ============================================================

def cleanup_directory(
    directory
):

    try:

        shutil.rmtree(
            directory,
            ignore_errors=True
        )

    except Exception:
        pass

    gc.collect()


# ============================================================
# PPT UPLOAD
# ============================================================

st.subheader("📂 Upload PowerPoint")

uploaded_file = st.file_uploader(
    "PPTX file upload karo — maximum 300 MB",
    type=["pptx"]
)


if uploaded_file:

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.write(
        f"📦 PPT Size: **{file_size_mb:.2f} MB**"
    )

    # ------------------------------------------
    # Size check
    # ------------------------------------------

    if file_size_mb > MAX_PPT_SIZE_MB:

        st.error(
            f"❌ File {MAX_PPT_SIZE_MB} MB se badi hai."
        )

        st.stop()

    # ------------------------------------------
    # Start button
    # ------------------------------------------

    start = st.button(
        "🚀 Start Professional Processing",
        type="primary",
        use_container_width=True
    )

    if start:

        # ==========================================
        # TEMP WORK DIRECTORY
        # ==========================================

        work_dir = tempfile.mkdtemp(
            prefix="ppt_ai_"
        )

        input_ppt = os.path.join(
            work_dir,
            "input.pptx"
        )

        output_ppt = os.path.join(
            work_dir,
            "professional_output.pptx"
        )

        try:

            # ======================================
            # SAVE PPT
            # ======================================

            with st.status(
                "📥 PPT save ho rahi hai...",
                expanded=True
            ) as status:

                save_uploaded_file(
                    uploaded_file,
                    input_ppt
                )

                status.update(
                    label="✅ PPT successfully save ho gayi",
                    state="complete"
                )

            # ======================================
            # FIND IMAGES
            # ======================================

            with st.status(
                "🔎 PPT ke andar images identify ho rahi hain...",
                expanded=True
            ) as status:

                media_paths = (
                    get_slide_media_references(
                        input_ppt
                    )
                )

                status.write(
                    f"Unique images found: "
                    f"**{len(media_paths)}**"
                )

                status.update(
                    label="✅ Images identify ho gayi",
                    state="complete"
                )

            if not media_paths:

                st.warning(
                    "PPT me supported images nahi mili."
                )

                st.stop()

            # ======================================
            # PROCESS IMAGES
            # ======================================

            st.subheader(
                "⚙️ AI Image Processing"
            )

            progress = st.progress(
                0
            )

            status_text = st.empty()

            replacements = {}

            success_count = 0
            failed_count = 0
            skipped_count = 0

            # --------------------------------------
            # Open PPT ZIP
            # --------------------------------------

            with zipfile.ZipFile(
                input_ppt,
                "r"
            ) as ppt_zip:

                total_images = len(
                    media_paths
                )

                for index, media_path in enumerate(
                    media_paths,
                    start=1
                ):

                    status_text.write(
                        f"🖼️ Processing "
                        f"{index}/{total_images}: "
                        f"{Path(media_path).name}"
                    )

                    ok, result = (
                        process_one_image(
                            client=client,
                            provider=provider,
                            ppt_zip=ppt_zip,
                            media_path=media_path,
                            work_dir=work_dir
                        )
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

                        st.warning(
                            f"⚠️ Failed: "
                            f"{Path(media_path).name}\n\n"
                            f"{result}"
                        )

                    progress.progress(
                        index / total_images
                    )

                    # ----------------------------------
                    # Memory cleanup
                    # ----------------------------------

                    gc.collect()

            status_text.write(
                "✅ AI image processing complete"
            )

            # ======================================
            # BUILD FINAL PPT
            # ======================================

            with st.status(
                "📦 Final PPT ban rahi hai...",
                expanded=True
            ) as status:

                build_final_ppt(
                    original_ppt=input_ppt,
                    output_ppt=output_ppt,
                    replacements=replacements
                )

                status.update(
                    label="✅ Final PPT ready",
                    state="complete"
                )

            # ======================================
            # FINAL RESULT
            # ======================================

            final_size_mb = (
                os.path.getsize(
                    output_ppt
                )
                /
                (1024 * 1024)
            )

            st.success(
                "🎉 Processing Complete!"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Images Processed",
                    success_count
                )

            with col2:

                st.metric(
                    "Failed",
                    failed_count
                )

            with col3:

                st.metric(
                    "Final PPT",
                    f"{final_size_mb:.1f} MB"
                )

            # ======================================
            # DOWNLOAD
            # ======================================

            st.subheader(
                "⬇️ Download"
            )

            with open(
                output_ppt,
                "rb"
            ) as download_file:

                st.download_button(
                    label="⬇️ Download Professional PPT",
                    data=download_file,
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
                "Original PPT ke slide layouts, positions "
                "aur slide XML ko modify nahi kiya gaya. "
                "Sirf supported raster images ko AI-edited "
                "images se replace kiya gaya hai."
            )

        except Exception as e:

            st.error(
                "❌ Processing error"
            )

            st.exception(
                e
            )
