from typing import Annotated

from fastapi import FastAPI, File, UploadFile

# from tensorflow import keras

app = FastAPI()

# model = keras.models.load_model("models/model.h5")

@app.get("/test")
def read_root():
    return {"Hello": "World"}

@app.post("/files")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
