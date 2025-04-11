import streamlit as st
import time
import easyocr
import mss
import cv2
import numpy as np
from PIL import Image
import os
import torch
from PyPDF2 import PdfReader
from sklearn.cluster import KMeans
from transformers import pipeline

# Conditional import of pyautogui
try:
    if os.environ.get("DISPLAY"):
        import pyautogui
        pyautogui_available = True
    else:
        pyautogui_available = False
except ImportError:
    pyautogui_available = False

# Initialize OCR and summarizer
reader = easyocr.Reader(['en'])
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def capture_active_screen():
    """Captures the primary screen"""
    time.sleep(2)
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save("dashboard.png")
        return img

def detect_active_filter(image_path):
    """Detects filters using OCR"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_results = reader.readtext(image_path)
    extracted_text = [text[1] for text in ocr_results]
    return extracted_text[0] if extracted_text else "Unknown"

def analyze_dashboard_image(image_path):
    """Extract and analyze dashboard area"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dashboard_img = img
    if contours:
        dashboard_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(dashboard_contour)
        dashboard_img = img[y:y+h, x:x+w]

    ocr_results = reader.readtext(dashboard_img)
    extracted_text = [text[1] for text in ocr_results]
    extracted_numbers = [word for text in extracted_text for word in text.split() if word.replace(",", "").replace(".", "").isdigit()]

    description = (
        f"Dashboard contains charts/tables. "
        f"Detected text: {'; '.join(extracted_text)}. "
        f"Numbers: {', '.join(extracted_numbers)}. "
    )
    return description

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def get_active_screen():
    if not pyautogui_available:
        return None

    mouse_x, mouse_y = pyautogui.position()
    with mss.mss() as sct:
        monitors = sct.monitors[1:]
        for monitor in monitors:
            if (monitor["left"] <= mouse_x < monitor["left"] + monitor["width"] and
                monitor["top"] <= mouse_y < monitor["top"] + monitor["height"]):
                return monitor
        return sct.monitors[1]




