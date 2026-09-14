import streamlit as st
import os
import gc
import time
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
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="PPT Professional Marking AI",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_PPT_SIZE_MB = 300

# Image Gemini/OpenAI ko bhejne se pehle maximum side
MAX_AI_SIDE = 1280

JPEG_QUALITY = 82
FINAL_JPEG_QUALITY = 92

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# MODELS
# ============================================================

# IMPORTANT:
# Stable model - preview model nahi
GEMINI_MODEL = "gemini-2.5-flash-image"

OPENAI_MODEL = "gpt-image-1"


# ============================================================
# AI PROMPT
# ============================================================

AI_PROMPT = """
Edit this real field-survey / recce photograph.

The image may contain a rough black hand-drawn marking,
scribble, circle, irregular line, or freehand outline that
is already BAKED INTO THE PHOTO PIXELS.

TASK:

Remove the rough marking and replace it with one clean,
professional rectangular marking around the SAME target.

STRICT RULES:

1. Find the rough hand-drawn marking.

2. Remove ONLY that rough marking.

3. Reconstruct the photograph underneath the removed marking
   using surrounding pixels, textures, colors, lighting,
   perspective and scene geometry.

4. The rough marking indicates the target area.
   Use that SAME area as the target.

5. Draw ONE clean professional rectangle around the SAME target.

6. Rectangle must have four straight sides.

7. Rectangle must be a thin BLACK outline.

8. Rectangle must have NO fill.

9. No circle.

10. No oval.

11. No scribble.

12. No arrow.

13. No glow.

14. No shadow.

15. No decorative effect.

16. Do not move the rectangle to another object.

17. Keep the rectangle close to the original rough marking.

18. Do not crop the image.

19. Do not change the image aspect ratio.

20. Do not change the composition.

21. Preserve people.

22. Preserve vehicles.

23. Preserve buildings.

24. Preserve signs and boards.

25. Preserve existing text.

26. Do not invent or rewrite text.

27. Preserve colors.

28. Preserve lighting and shadows.

29. Preserve perspective.

30. Do not add unrelated objects.

31. Do not remove unrelated objects.

32. Do not modify areas outside the rough marking unless
    absolutely necessary to reconstruct the background.

33. If there is no rough marking, return the image essentially
    unchanged.

34. The final result should look like a professional
    site-recce photograph.

35. Return ONLY the edited image.
"""


# ============================================================
# HELPERS
# ============================================================

def cleanup_memory():
    gc.collect()


def is_quota_zero_error(error_text):
    text = str(error_text).lower()

    return (
        "limit: 0" in text
        or "quota_limit_value: 0" in text
        or "free_tier" in text and "limit: 0" in text
    )


def is_retryable_error(error_text):
    text = str(error_text).lower()

    return (
        "429" in text
        or "resource_exhausted" in text
        or "rate limit" in text
        or "too many requests" in text
        or "503" in text
        or "unavailable" in text
    )


# ============================================================
# API CLIENT
# ============================================================

def create_client(provider, api_key):

    if provider == "Google Gemini":

        return genai.Client(
            api_key=api_key
        )

    else:

        return OpenAI(
            api_key=api_key
        )


# ============================================================
# SAVE UPLOADED PPT TO DISK
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
# FIND PPT IMAGES
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

            rel_filename = Path(
                rels_name
            ).name

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

        if img.mode != "RGB":

            img = img.convert(
                "RGB"
            )

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

        img.save(
            ai_path,
            format="JPEG",
            quality=JPEG_QUALITY
        )

    cleanup_memory()

    return (
        original_width,
        original_height
    )


# ============================================================
# GEMINI IMAGE EDIT
# ============================================================

def process_with_gemini(
    client,
    ai_path,
    output_path
):

    MAX_RETRIES = 3

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            with open(
                ai_path,
                "rb"
            ) as image_file:

                image_bytes = (
                    image_file.read()
                )

            try:

                response = (
                    client.models.generate_content(

                        model=GEMINI_MODEL,

                        contents=[
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/jpeg"
                            ),
                            AI_PROMPT
                        ],

                        config=(
                            types.GenerateContentConfig(
                                response_modalities=[
                                    "IMAGE"
                                ]
                            )
                        )
                    )
                )

            finally:

                del image_bytes

                cleanup_memory()

            generated_data = None

            if response.parts:

                for part in response.parts:

                    if (
                        part.inline_data
                        is not None
                    ):

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
            ) as output:

                output.write(
                    generated_data
                )

            del generated_data
            del response

            cleanup_memory()

            return True, None

        except Exception as e:

            error_text = str(e)

            # -----------------------------------------
            # Quota is literally ZERO
            # -----------------------------------------

            if is_quota_zero_error(
                error_text
            ):

                raise RuntimeError(
                    "Gemini image-generation quota "
                    "ZERO hai.\n\n"
                    "Ye API key format ki problem nahi hai. "
                    "Google project ke current tier me "
                    "image model ka quota 0 hai.\n\n"
                    "Google AI Studio me billing/paid tier "
                    "enable karke dobara try karo."
                )

            # -----------------------------------------
            # Temporary 429 / 503
            # -----------------------------------------

            if (
                is_retryable_error(
                    error_text
                )
                and attempt < MAX_RETRIES - 1
            ):

                wait_seconds = (
                    8 * (2 ** attempt)
                )

                time.sleep(
                    wait_seconds
                )

                cleanup_memory()

                continue

            raise


# ============================================================
# OPENAI IMAGE EDIT
# ============================================================

def process_with_openai(
    client,
    ai_path,
    output_path
):

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

    image_result = (
        result.data[0]
    )

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
    ) as output:

        output.write(
            image_bytes
        )

    del image_bytes
    del result

    cleanup_memory()


# ============================================================
# RESTORE ORIGINAL SIZE
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
            quality=FINAL_JPEG_QUALITY
        )

    cleanup_memory()


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

        # ==============================================
        # Extract ONLY this image
        # ==============================================

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

        # ==============================================
        # Verify image
        # ==============================================

        try:

            with Image.open(
                original_path
            ) as test_image:

                test_image.verify()

        except Exception:

            return (
                False,
                "invalid_image"
            )

        # ==============================================
        # Prepare AI image
        # ==============================================

        (
            original_width,
            original_height
        ) = prepare_image_for_ai(
            original_path,
            ai_path
        )

        # ==============================================
        # AI EDIT
        # ==============================================

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

        # ==============================================
        # Restore original dimensions
        # ==============================================

        restore_original_size(
            generated_path,
            final_path,
            original_width,
            original_height
        )

        # ==============================================
        # Remove temporary files
        # ==============================================

        for file_path in [
            original_path,
            ai_path,
            generated_path
        ]:

            try:

                if os.path.exists(
                    file_path
                ):

                    os.remove(
                        file_path
                    )

            except Exception:
                pass

        cleanup_memory()

        return (
            True,
            final_path
        )

    except Exception as e:

        for file_path in [
            original_path,
            ai_path,
            generated_path,
            final_path
        ]:

            try:

                if os.path.exists(
                    file_path
                ):

                    os.remove(
                        file_path
                    )

            except Exception:
                pass

        cleanup_memory()

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
    ) as source_zip:

        with zipfile.ZipFile(
            output_ppt,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as output_zip:

            for info in source_zip.infolist():

                filename = info.filename

                replacement = (
                    replacements.get(
                        filename
                    )
                )

                # ==========================================
                # Replace processed image
                # ==========================================

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

                        with output_zip.open(
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

                # ==========================================
                # Copy original PPT content
                # ==========================================

                else:

                    with source_zip.open(
                        info,
                        "r"
                    ) as source:

                        with output_zip.open(
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

    cleanup_memory()


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

    cleanup_memory()


# ============================================================
# API PROVIDER
# ============================================================

st.subheader("🤖 AI Provider")

provider = st.radio(
    "AI select karo:",
    [
        "Google Gemini",
        "OpenAI"
    ],
    horizontal=True
)


# ============================================================
# API KEY
# ============================================================

if provider == "Google Gemini":

    st.subheader(
        "🔑 Gemini API Key"
    )

    st.caption(
        "AQ... wali Gemini Authorization API key bhi "
        "yahan paste kar sakte ho."
    )

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AQ....",
        help="Google AI Studio ki Gemini API key."
    )

else:

    st.subheader(
        "🔑 OpenAI API Key"
    )

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-....",
        help="OpenAI Platform ki API key."
    )


if not api_key:

    st.info(
        "Pehle API key enter karo."
    )

    st.stop()


# ============================================================
# CREATE CLIENT
# ============================================================

try:

    client = create_client(
        provider,
        api_key
    )

    st.success(
        f"✅ {provider} API ready"
    )

except Exception as e:

    st.error(
        f"❌ API setup error: {e}"
    )

    st.stop()


# ============================================================
# PPT UPLOAD
# ============================================================

st.subheader(
    "📂 Upload PowerPoint"
)

uploaded_file = st.file_uploader(
    "PPTX file upload karo — maximum 300 MB",
    type=["pptx"]
)


if uploaded_file:

    file_size_mb = (
        uploaded_file.size
        /
        (1024 * 1024)
    )

    st.write(
        f"📦 PPT Size: **{file_size_mb:.2f} MB**"
    )

    if file_size_mb > MAX_PPT_SIZE_MB:

        st.error(
            "❌ PPT 300 MB se badi hai."
        )

        st.stop()

    start = st.button(
        "🚀 Start Professional Processing",
        type="primary",
        use_container_width=True
    )

    if start:

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

            # ==========================================
            # SAVE PPT
            # ==========================================

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

            # ==========================================
            # FIND IMAGES
            # ==========================================

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
                    "PPT me supported raster images nahi mili."
                )

                st.stop()

            # ==========================================
            # PROCESS
            # ==========================================

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

            total_images = len(
                media_paths
            )

            with zipfile.ZipFile(
                input_ppt,
                "r"
            ) as ppt_zip:

                for index, media_path in enumerate(
                    media_paths,
                    start=1
                ):

                    filename = Path(
                        media_path
                    ).name

                    status_text.write(
                        f"🖼️ Processing "
                        f"{index}/{total_images}: "
                        f"{filename}"
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

                        # ----------------------------------
                        # Special Gemini quota error
                        # ----------------------------------

                        if (
                            provider == "Google Gemini"
                            and
                            (
                                "quota" in str(
                                    result
                                ).lower()
                                or
                                "resource_exhausted"
                                in str(
                                    result
                                ).lower()
                            )
                        ):

                            st.error(
                                "❌ Gemini quota problem\n\n"
                                + str(result)
                            )

                            st.stop()

                        else:

                            st.warning(
                                f"⚠️ Failed: {filename}\n\n"
                                f"{result}"
                            )

                    progress.progress(
                        index / total_images
                    )

                    cleanup_memory()

            status_text.write(
                "✅ AI image processing complete"
            )

            # ==========================================
            # BUILD FINAL PPT
            # ==========================================

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

            # ==========================================
            # RESULT
            # ==========================================

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
                    "Processed",
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

            # ==========================================
            # DOWNLOAD
            # ==========================================

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
                "Slide layout aur slide XML ko modify nahi kiya gaya. "
                "Sirf supported raster images ko AI-edited images "
                "se replace kiya gaya hai."
            )

        except Exception as e:

            st.error(
                "❌ Processing error"
            )

            st.exception(
                e
            )
