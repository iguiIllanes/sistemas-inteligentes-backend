from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io

from utils.classifier import Classifier
from utils.segmentation import Segmentation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (change as needed)
    allow_headers=["*"],  # Allow all headers (change as needed)
)

@app.post("/test")
async def test():
    return FileResponse('test.png')

@app.post("/classifier")
async def test(file: UploadFile):
    
    classifier = Classifier("models/model.h5")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    prediction = classifier.make_prediction(image)

    if(prediction == 0):
        response = 'Es un perro'
    else:
        response = 'Es un gato'

    return {
        "prediction": response
    }

@app.post('/classifier-segmentation')
async def classifier_segmentation(file: UploadFile):
    classifier_segmentation = Segmentation('models/dogs_cats.h5')

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    prediction = classifier_segmentation.make_prediction(image)

    return {
        "prediction": prediction
    }
