import io
import gc
import json
import requests
import streamlit as st
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="PPT AI Box Fixer", layout="wide")

st.title("PPT Hand-Drawn Box AI Fixer")
st.write("Gemini 1.5 Flash + HuggingFace API based auto box straightener & color matcher.")

# Sidebar for API Keys
st.sidebar.header("API Configurations")
gemini_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
hf_token = st.sidebar.text_input("Enter HuggingFace Access Token:", type="password")

# HuggingFace Free LaMa Model URL
HF_API_URL = "https://api-inference.huggingface.co/models/lama-cleaner/lama"

def clean_via_hf_api(img_bytes, mask_bytes, token):
    """HuggingFace API ka use karke lines ko background se erase karna"""
    headers = {"Authorization": f"Bearer {token}"}
    files = {
        'image': ('image.png', img_bytes, 'image/png'),
        'mask': ('mask.png', mask_bytes, 'image/png')
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, files=files, timeout=30)
        if response.status_code == 200:
            return response.content
        return img_bytes
    except Exception:
        return img_bytes

def process_single_image(image_bytes, g_key, hf_tok):
    """Gemini Vision se Detection + HuggingFace se Inpainting + Straight Box Overlay"""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img_pil.size

        # Step 1: Gemini Vision API Call
        genai.configure(api_key=g_key)
        vision_model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = """
        Analyze this image for any hand-drawn markings or irregular boxes drawn over shop signs/objects.
        Respond strictly in valid JSON format with no additional text:
        {
          "has_mark": true/false,
          "color": [R, G, B],
          "box_percent": [ymin, xmin, ymax, xmax]
        }
        where box_percent contains boundary percentage coordinates from 0 to 100.
        """

        response = vision_model.generate_content([img_pil, prompt])
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)

        if not data.get("has_mark", False):
            return image_bytes

        # Bounding box percent to pixels conversion
        box = data["box_percent"]
        color = tuple(data.get("color", [0, 0, 0]))

        ymin = int((box[0] / 100) * h)
        xmin = int((box[1] / 100) * w)
        ymax = int((box[2] / 100) * h)
        xmax = int((box[3] / 100) * w)

        # Step 2: Mask Generation for Inpainting
        mask = Image.new("L", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask)
        # Expansion padding around line
        draw_mask.rectangle([max(0, xmin-10), max(0, ymin-10), min(w, xmax+10), min(h, ymax+10)], fill=255)

        mask_io = io.BytesIO()
        mask.save(mask_io, format="PNG")
        mask_bytes = mask_io.getvalue()

        # Step 3: Cloud Erasing via HuggingFace
        cleaned_bytes = clean_via_hf_api(image_bytes, mask_bytes, hf_tok)
        cleaned_pil = Image.open(io.BytesIO(cleaned_bytes)).convert("RGB")

        # Step 4: Straight Rectangular Box Redraw
        draw = ImageDraw.Draw(cleaned_pil)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

        output_io = io.BytesIO()
        cleaned_pil.save(output_io, format="PNG")
        return output_io.getvalue()

    except Exception:
        return image_bytes

# Main Streamlit UI Logic
uploaded_file = st.file_uploader("Upload PPT File (< 200MB)", type=["pptx"])

if uploaded_file is not None:
    if not gemini_key or not hf_token:
        st.warning("⚠️ Kripya Sidebar me Gemini Key aur HuggingFace Token dono daalein!")
    else:
        if st.button("Start Processing PPT"):
            prs = Presentation(uploaded_file)
            total_slides = len(prs.slides)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, slide in enumerate(prs.slides, start=1):
                status_text.text(f"Processing Slide {idx}/{total_slides}...")

                for shape in slide.shapes:
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        try:
                            img_bytes = shape.image.blob
                            fixed_bytes = process_single_image(img_bytes, gemini_key, hf_token)
                            shape.image.blob = fixed_bytes
                        except Exception:
                            pass

                progress_bar.progress(idx / total_slides)
                if idx % 5 == 0:
                    gc.collect()

            # Save processed PPT to buffer
            output_stream = io.BytesIO()
            prs.save(output_stream)
            output_stream.seek(0)

            st.success("✅ Presentation Processed Successfully!")
            st.download_button(
                label="Download Cleaned PPT",
                data=output_stream,
                file_name="straightened_boxes_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
