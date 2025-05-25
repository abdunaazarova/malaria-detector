from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = FastAPI()
model = tf.keras.models.load_model('Malaria.h5')

IMG_SIZE = 128
app.mount('/static', StaticFiles(directory='static'),name='static')

@app.get('/',response_class=HTMLResponse)
async def root():
    with open('static/index.html') as f:
        return f.read()
    
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))  
    image = np.array(image).astype(np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    pred = float(model.predict(image)[0][0])

    if pred > 0.5:
        label = 'Uninfected'
        confidence = pred
    else:
        label = 'Parasitized'
        confidence = 1 - pred
    return {
        'prediction':label,
        'confidence': round(confidence*100,2)
    }
























































