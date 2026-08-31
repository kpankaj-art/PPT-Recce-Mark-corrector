import cv2
import numpy as np

def fix_any_color_drawn_box(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Image Noise & Texture ko smooth karna
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Edges detect karna (Color chahe koi bhi ho)
    edges = cv2.Canny(blurred, 30, 150)
    
    # Lines ko thoda connect/Thick karna taaki gaps fill ho jayein
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # 3. Contours (Shapes) find karna
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_area = 0
    mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Background Noise ignore karne ke liye Area thresholding
        if area > 1500 and area < (img.shape[0] * img.shape[1] * 0.8):
            # Shape Boundary check karna
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Agar stroke ek closed shape (3 se 8 corners) bana raha hai
            if area > max_area:
                max_area = area
                best_rect = cv2.boundingRect(cnt)
                cv2.drawContours(mask, [cnt], -1, 255, thickness=12)

    if best_rect is not None:
        x, y, w, h = best_rect
        
        # 4. Inpainting: Har color ki line ko erase kar background blend karna
        clean_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        
        # 5. Fixed Standard Color (e.g. Green ya Red) ka Perfect Straight Box draw karna
        cv2.rectangle(clean_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        cv2.imwrite(output_path, clean_img)
        return True

    return False
