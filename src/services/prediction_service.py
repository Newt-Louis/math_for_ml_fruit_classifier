from fastapi import UploadFile
from PIL import Image
from src.utils.image_utils import read_image_and_resize
from src.models.model_loader import get_model
from src.core.config import settings

class PredictionService:

    @staticmethod
    async def predict(file: UploadFile) -> str:
        # Step 1: Load & preprocess image
        image: Image.Image = await read_image_and_resize(file, size=settings.image_size)

        # Step 2: Run inference
        model = get_model()
        label = model.predict(image)

        # Step 3: Post-process & return
        return label
