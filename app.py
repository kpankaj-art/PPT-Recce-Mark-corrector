import io
import os
import json
import warnings
import streamlit as st
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from google import genai

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PPT Auto Cleaner", page_icon="🎨", layout="centered")

st.title("🎨 PPT Image Clean-up Tool")
st.write("Upload your PPTX file to auto-detect marks and restore clean borders.")

# Gemini API Key Secrets se uthayega
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)

if not GEMINI_API_KEY:
    st.error("Gemini API Key missing! Please configure Secrets in Streamlit.")
    st.stop()

client = get_client(GEMINI_API_KEY)

def detect_mark(pil_image):
    prompt = """
    Analyze this image for hand-drawn irregular lines, circles, or rough boxes over shop boards/objects.
    Respond strictly in JSON format with:
    {
      "has_mark": true,
      "color": [R, G, B],
      "box_percent": [ymin, xmin, ymax, xmax]
    }
    where box_percent contains percentage coordinates from 0 to 100.
    """
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[pil_image, prompt]
        )
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)

        if not data.get("has_mark", False):
            return None, None

        box = data["box_percent"]
        color = tuple(data.get("color", [255, 0, 0]))
        return box, color
    except Exception:
        return None, None

def process_presentation(uploaded_file):
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
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    w, h = img_pil.size

                    box, color = detect_mark(img_pil)
                    if box:
                        ymin = int((box[0] / 100) * h)
                        xmin = int((box[1] / 100) * w)
                        ymax = int((box[2] / 100) * h)
                        xmax = int((box[3] / 100) * w)

                        # Clean Box Overwrite
                        draw = ImageDraw.Draw(img_pil)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

                        output = io.BytesIO()
                        img_pil.save(output, format="PNG")
                        shape.image.blob = output.getvalue()
                except Exception:
                    pass
        progress_bar.progress(idx / total_slides)

    status_text.text("Processing Complete!")
    
    ppt_out = io.BytesIO()
    prs.save(ppt_out)
    ppt_out.seek(0)
    return ppt_out

uploaded_file = st.file_uploader("Upload PowerPoint File (.pptx)", type=["pptx"])

if uploaded_file is not None:
    if st.button("Fix & Clean PPT", type="primary"):
        with st.spinner("AI Processing PPT..."):
            fixed_ppt = process_presentation(uploaded_file)
            st.success("PPT Successfully Fixed!")
            
            st.download_button(
                label="📥 Download Fixed PPT",
                data=fixed_ppt,
                file_name=f"fixed_{uploaded_file.name}",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
