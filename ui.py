import streamlit as st
import requests
import numpy as np
from PIL import Image
import cv2

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI.", use_column_width=True)
    
    # Convert to grayscale and resize to match input
    image = np.array(image.convert("L"))
    image = cv2.resize(image, (256, 256))
    
    # Send to FastAPI
    response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file.getvalue()})
    prediction = np.array(response.json()["segmentation"])
    
    st.image(prediction.squeeze(), caption="Segmentation Result", use_column_width=True)
