import streamlit as st
import cv2
import numpy as np
from pptx import Presentation
from pptx.util import Inches
import io
from PIL import Image

def fix_drawn_box_in_image(image_bytes):
    # Image bytes ko OpenCV format me convert karna
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return image_bytes, False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_area = 0
    mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500 and area < (img.shape[0] * img.shape[1] * 0.8):
            if area > max_area:
                max_area = area
                best_rect = cv2.boundingRect(cnt)
                cv2.drawContours(mask, [cnt], -1, 255, thickness=12)

    if best_rect is not None:
        x, y, w, h = best_rect
        clean_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        # Green Perfect Straight Box Draw Karna
        cv2.rectangle(clean_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # Fixed Image ko PNG bytes me convert karke return karna
        _, encoded_img = cv2.imencode('.png', clean_img)
        return encoded_img.tobytes(), True

    return image_bytes, False

def process_ppt(ppt_bytes):
    prs = Presentation(io.BytesIO(ppt_bytes))
    modified = False

    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type == 13: # 13 = Picture shape type
                image_stream = shape.image.blob
                fixed_image_bytes, is_modified = fix_drawn_box_in_image(image_stream)
                
                if is_modified:
                    modified = True
                    # Slide me purani image ko naye fixed image se replace karna
                    new_img_stream = io.BytesIO(fixed_image_bytes)
                    left, top, width, height = shape.left, shape.top, shape.width, shape.height
                    
                    # Purani image shape ko remove karke naye image ko add karna
                    sp = shape._element
                    sp.getparent().remove(sp)
                    slide.shapes.add_picture(new_img_stream, left, top, width, height)

    out_stream = io.BytesIO()
    prs.save(out_stream)
    out_stream.seek(0)
    return out_stream.getvalue(), modified

# --- STREAMLIT UI ---
st.set_page_config(page_title="PPT Recce Mark Corrector", layout="centered")

st.title("PPT Recce Mark Corrector")
st.write("PPT upload karein, ye tool hand-drawn (tedhe-medhe) boxes ko automatically straight green boxes me fix kar dega.")

uploaded_ppt = st.file_uploader("Apni PowerPoint (.pptx) file choose karein", type=["pptx"])

if uploaded_ppt is not None:
    if st.button("Process PPT", type="primary"):
        with st.spinner("Images scan aur clean ho rahi hain..."):
            processed_ppt_bytes, status = process_ppt(uploaded_ppt.read())
            
            if status:
                st.success("PPT Successfully Process Ho Gayi Hai!")
                st.download_button(
                    label="Download Fixed PPT",
                    data=processed_ppt_bytes,
                    file_name=f"Fixed_{uploaded_ppt.name}",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
            else:
                st.warning("PPT ke images me koi hand-drawn marker box detect nahi hua.")
