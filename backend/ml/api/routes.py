from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.ml.inference.predictor import PredictorError, ridgevision_predictor

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "RidgeVision AI"}


@router.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = False) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a fingerprint image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        return ridgevision_predictor.predict(image_bytes, explain=explain)
    except PredictorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
