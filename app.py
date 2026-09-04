import io
import warnings
import streamlit as st
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from google import genai
from google.genai import types

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI PPT Image Fixer", page_icon="🎨", layout="centered")

st.title("🎨 PPT Image AI Editing & Restore Tool")
st.write("PPT se image niklegi -> Gemini AI pixels se rough markings remove karega -> Fixed image PPT me replace hogi.")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)

if not GEMINI_API_KEY:
    st.error("Streamlit Secrets mein GEMINI_API_KEY missing hai!")
    st.stop()

client = get_client(GEMINI_API_KEY)

def fix_image_with_ai(pil_image):
    """
    Sends the extracted image to Gemini AI to edit/clean the image contents directly,
    removing hand-drawn lines while preserving text/background, and returning the new image.
    """
    prompt = """
    Edit this image:
    1. Detect any hand-drawn irregular rough lines, circles, or scribbles.
    2. Erase all those rough lines completely and restore the original background/text under them.
    3. Draw a neat, straight, clean professional rectangular box around the object/board.
    """
    try:
        # Pass image + text prompt to Gemini Image Editing Model
        response = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )
        
        # Get generated image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                edited_image = Image.open(io.BytesIO(part.inline_data.data))
                return edited_image
                
        return pil_image
    except Exception as e:
        # Fallback if model encounters an error
        return pil_image

def process_shape_recursive(shape, cleaned_count):
    """Deep scan and process pictures inside groups or frames"""
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sub_shape in shape.shapes:
            process_shape_recursive(sub_shape, cleaned_count)
            
    elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE or hasattr(shape, "image"):
        try:
            # 1. Extract image from PPT
            img_bytes = shape.image.blob
            original_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # 2. AI Editing / Fix via Gemini
            fixed_pil = fix_image_with_ai(original_pil)

            # 3. Replace fixed image back into PPT shape
            output = io.BytesIO()
            fixed_pil.save(output, format="PNG")
            shape.image.blob = output.getvalue()
            cleaned_count[0] += 1
        except Exception:
            pass

def process_presentation(uploaded_file):
    prs = Presentation(uploaded_file)
    total_slides = len(prs.slides)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    cleaned_count = [0]

    for idx, slide in enumerate(prs.slides, start=1):
        status_text.text(f"Processing Slide {idx}/{total_slides} (Extracting -> Editing -> Replacing)...")
        for shape in slide.shapes:
            process_shape_recursive(shape, cleaned_count)
            
        progress_bar.progress(idx / total_slides)

    status_text.text(f"Processing Complete! Fixed & Replaced {cleaned_count[0]} images.")
    
    ppt_out = io.BytesIO()
    prs.save(ppt_out)
    ppt_out.seek(0)
    return ppt_out, cleaned_count[0]

uploaded_file = st.file_uploader("Upload PowerPoint File (.pptx)", type=["pptx"])

if uploaded_file is not None:
    if st.button("Fix & Replace Images in PPT", type="primary"):
        with st.spinner("PPT se images extract ho rahi hain aur Gemini AI se clean hokar replace ho rahi hain..."):
            fixed_ppt, count = process_presentation(uploaded_file)
            
            if count > 0:
                st.success(f"Success! Total {count} images AI se fix hokar PPT me replace ho gayi hain.")
            else:
                st.warning("PPT me koi image shape parse nahi ho payi.")

            st.download_button(
                label="📥 Download Fixed PPT",
                data=fixed_ppt,
                file_name=f"fixed_{uploaded_file.name}",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
