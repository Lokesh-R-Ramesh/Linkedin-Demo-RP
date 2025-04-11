import streamlit as st
import time
import easyocr
import mss
import cv2
import numpy as np
from PIL import Image
import os
if os.environ.get("DISPLAY"):
    import pyautogui
import torch
from PyPDF2 import PdfReader
from sklearn.cluster import KMeans
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import BedrockChat
import boto3
import json
import boto3
from langchain_community.chat_models import BedrockChat
import pandas as pd
import snowflake.connector
import zipfile
import os
import json
import shutil
import gzip
import gzip
import json
import pandas as pd 


# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Load Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def capture_active_screen():
    """Captures the active screen where the mouse is positioned."""
    time.sleep(2)
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save("dashboard.png")
        return img

def detect_active_filter(image_path):
    """Detects active filters using OCR and color analysis."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_results = reader.readtext(image_path)
    extracted_text = [text[1] for text in ocr_results]
    
    return extracted_text[0] if extracted_text else "Unknown"

def analyze_dashboard_image(image_path):
    """Extracts text and numbers from the dashboard area, detects active filters, and prepares analysis."""
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect the dashboard area using edge detection and contour detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest rectangular contour is the dashboard
    dashboard_contour = max(contours, key=cv2.contourArea, default=None)
    if dashboard_contour is not None:
        x, y, w, h = cv2.boundingRect(dashboard_contour)
        dashboard_img = img[y:y+h, x:x+w]
    else:
        dashboard_img = img  # Fallback to full image if no contour detected
    
    # OCR on the detected dashboard area
    ocr_results = reader.readtext(dashboard_img)
    extracted_text = [text[1] for text in ocr_results]
    extracted_numbers = [word for text in extracted_text for word in text.split() if word.replace(",", "").replace(".", "").isdigit()]
    
    # Detect active filter (Assuming a placeholder function detect_active_filter)
    #active_filter = detect_active_filter(image_path)
    
    description = (f"Dashboard contains charts/tables. "
                   f"Detected text: {'; '.join(extracted_text)}. "
                   f"Numbers: {', '.join(extracted_numbers)}. ")
                   #f"Active filter: {active_filter}.")
    
    return description

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    
def get_active_screen():
    """Finds the monitor where the mouse is currently located."""
    mouse_x, mouse_y = pyautogui.position()

    with mss.mss() as sct:
        monitors = sct.monitors[1:]
        for monitor in monitors:
            if (monitor["left"] <= mouse_x < monitor["left"] + monitor["width"] and
                monitor["top"] <= mouse_y < monitor["top"] + monitor["height"]):
                return monitor

    return sct.monitors[1]

def Metadata_File_Extractor(pbit_file_path):
    extract_folder = r"C:\Users\LokeshRamesh\Documents\co_10 training\LLM\Power Bi Smart Bot\Extract"
    if os.path.exists(extract_folder):
        shutil.rmtree(extract_folder)  # Clean up previous extraction
    os.makedirs(extract_folder)

    with zipfile.ZipFile(pbit_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    data_model_path = None
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if file == "DataModelSchema":
                data_model_path = os.path.join(root, file)
                break
    with open(data_model_path, "rb") as f:
        data = f.read(20)  # Read first 20 bytes
    
    with open(data_model_path, "rb") as f:
        raw_data = f.read()

# Try to decode as UTF-16 and handle errors if it's not in that encoding
    try:
        decoded_text = raw_data.decode("utf-16")
    except UnicodeDecodeError:
        decoded_text = raw_data.decode("utf-8")  # Fallback to UTF-8 if needed

    # Convert to a JSON dictionary
    json_data = json.loads(decoded_text)

    # Convert JSON to Pandas DataFrame
    df = pd.DataFrame(json_data)  # Ensure json_data is in a tabular format
    return df

def Metadata_Extractor(file_path,llm):
    # Load JSON metadata safely
    metadata = Metadata_File_Extractor(file_path)
    dax_df = pd.DataFrame()

    # Extract tables from metadata
    tables = metadata.get("model", {}).get("tables", [])

    Relation_df = pd.DataFrame(columns=["fromTable", "FromColumn","toTable", "ToColumn"])

# Extract tables from metadata
    relationships = metadata.get("model", {}).get("relationships", [])

    for i in range(len(relationships)):
        Relation_df = pd.concat(
            [Relation_df, pd.DataFrame([{"fromTable": relationships[i]["fromTable"], "FromColumn": relationships[i]["fromColumn"], "toTable": relationships[i]["toTable"], "ToColumn": relationships[i]["toColumn"]}])], 
            ignore_index=True
        )

    for table in tables:
        if "measures" in table:
            dax_df = pd.concat([dax_df, pd.DataFrame(table["measures"])], ignore_index=True)

    # If there are no measures, return an empty DataFrame
    if dax_df.empty:
        return pd.DataFrame(columns=["Dax Name", "Formatted DAX"])

    # Select and rename relevant columns
    measure_table = dax_df[["name", "expression"]].rename(
        columns={"expression": "DAX", "name": "Dax Name"}
    )

    # Function to determine data type
    def get_dtype(value):
        if isinstance(value, list):
            return "list"
        elif isinstance(value, str):
            return "string"
        return type(value).__name__

    # Function to extract the correct formula
    def extract_formula(expr):
        if isinstance(expr, list) and len(expr) > 1:
            return expr[1]  # Return second element if list has more than one item
        return expr  # Otherwise, return as is

    # Apply transformations correctly
    measure_table["Dtype"] = measure_table["DAX"].apply(get_dtype)
    measure_table["Formatted DAX"] = measure_table["DAX"].apply(extract_formula)

    PowerBI_Table_DF = pd.DataFrame(columns=['Table Name', 'Column Name'])
    tables = metadata["model"]['tables']
    for table in tables:
        if 'columns' in table and "isHidden" not in table:
            for column in table['columns']:
               PowerBI_Table_DF = pd.concat([PowerBI_Table_DF, pd.DataFrame({'Table Name': [table['name']], 'Column Name': [column['name']], 'Data Type': [column['dataType']], "Summaries" : [column["summarizeBy"]]})], ignore_index=True)

    Source_df = pd.DataFrame(columns=['Table Name', 'expression'])
    metadata_list = metadata["model"]["tables"]

    for i in range(len(metadata_list)):
        if "isHidden" not in metadata_list[i]:  # Check if "isHidden" key is absent
            partitions = metadata_list[i].get("partitions", [])  # Get partitions list safely

            if partitions:  # Ensure partitions exist and are not empty
                partition_name = partitions[0].get("name", None)
                expression = partitions[0].get("source", {}).get("expression", None)

                Source_df = pd.concat([Source_df, pd.DataFrame({"Table Name": [partition_name], "expression": [expression]})], ignore_index=True)
    
    def process_element(value):
        if isinstance(value, list):  # If it's a list
            return value[1] if len(value) > 1 else None  # Get 2nd element if possible
        elif isinstance(value, str):  # If it's a string
            return value
        else:
            return None  # Return None for other types
        
    def get_dtype(value):
        return "list" if isinstance(value, list) else "string" if isinstance(value, str) else type(value).__name__

# Optimized function to extract the second element if it's a list
    def extract_formula(row):
        expr = row.get("expression")
        return expr[1] if isinstance(expr, list) and len(expr) > 1 else expr

# Efficiently apply transformations
    Source_df["Dtype"] = Source_df["expression"].map(get_dtype)  # Uses .map() instead of apply()
    Source_df["Formula"] = Source_df.apply(extract_formula, axis=1)

    # Convert processed data into a DataFrame
    Source_df = Source_df[["Table Name","Formula"]]

    
    def formula_extractor(value):
        if pd.isna(value) or not isinstance(value, str):  # Handle missing or invalid values
            return None

        prompt = (
            f"Extract and return only the SQL script or DAX or Python formula from the following input. "
            f"Do not include any explanations, comments, or additional text. Output only the clean, formatted SQL or DAX:\n\n{value.strip()}"
        )

        output = llm.invoke(prompt).content.strip()  # Ensure clean output
        return output
    def source_detection(value):
        if pd.isna(value) or not isinstance(value, str):  # Handle missing or invalid values
            return None

        prompt = (
            f"Extract the source name from the script below. "
            f"If it's a DAX formula, return 'Calculated Table'. "
            f"Otherwise, return only the source name (e.g., 'Snowflake'). "
            f"Do not include any extra text:\n\n{value.strip()}"
        )

        output = llm.invoke(prompt).content.strip()  # Ensure clean output
        return output
    def generate_transformation_script(value):
        if pd.isna(value) or not isinstance(value, str):  # Handle missing or invalid values
            return None

        prompt = (
                f"Explain how the given data is imported into Power BI in short form. "
                f"Provide details on the data sources, connections, transformations, and loading process. "
                f"Ensure the response covers how Power BI fetches, processes, and integrates the data into its model:\n\n{value.strip()}"
            )

        output = llm.invoke(prompt).content.strip()  # Ensure clean output
        return output
    Source_df["Script"] = Source_df["Formula"].map(formula_extractor)
    Source_df["Source data"] = Source_df["Formula"].map(source_detection)
    Source_df["Explain"] = Source_df["Formula"].map(generate_transformation_script)
    Source_df=Source_df[["Table Name","Source data","Script","Explain"]]

    return measure_table[["Dax Name", "Formatted DAX"]] , PowerBI_Table_DF , Source_df, Relation_df
