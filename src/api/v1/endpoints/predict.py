from fastapi import APIRouter, UploadFile, File, HTTPException
from src.schemas.prediction_schema import PredictionResponse
from src.services.prediction_service import PredictionService

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        label = await PredictionService.predict(file)
        return {"label": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
