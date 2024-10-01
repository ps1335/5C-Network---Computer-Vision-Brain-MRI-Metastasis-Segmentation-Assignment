from fastapi import FastAPI, File, UploadFile
import uvicorn
import cv2
import numpy as np
from models import attention_unet

app = FastAPI()

# Load the trained model (example: Attention U-Net)
model = attention_unet(input_shape=(256, 256, 1))  # Adjust input shape
model.load_weights('best_attention_unet_model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=-1) / 255.0
    
    prediction = model.predict(np.expand_dims(image, axis=0))
    prediction = (prediction > 0.5).astype(np.uint8)
    
    return {"segmentation": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
