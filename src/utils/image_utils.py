from PIL import Image
import io
from fastapi import UploadFile

async def read_image_and_resize(file: UploadFile, size=(224, 224)) -> Image.Image:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return image.resize(size)
