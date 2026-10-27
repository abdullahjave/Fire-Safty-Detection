
import uuid
import logging
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import pytz
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from sqlalchemy.orm import Session
from ultralytics import YOLO

from app.api.predict_fire.schema import Detection, PredictFireSchema
from app.api.predict_fire.crud import create_detection_log
from app.db.database import get_db


# ==================================================
# 1. Router and Logger
# ==================================================

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# ==================================================
# 2. Project Paths
# ==================================================

# Current file:
# backend/app/api/predict_fire/router.py

# Backend app directory:
# backend/app
APP_DIR = Path(__file__).resolve().parents[2]

# YOLO model path:
# backend/app/assets/best.pt
MODEL_PATH = APP_DIR / "assets" / "best.pt"

# Temporary images directory:
# backend/temp
TEMP_DIR = APP_DIR.parent / "temp"

# Detection result images directory:
# backend/app/log
LOG_DIR = APP_DIR / "log"

# Create directories if missing
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. Load YOLO Model Once
# ==================================================

try:
    logger.info("Loading YOLO Fire Detection model...")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    logger.info("YOLO model loaded successfully!")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Model classes: {model.names}")

except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise


# ==================================================
# 4. Helper Functions
# ==================================================

def allowed_file(filename: str) -> bool:
    """
    Check whether the uploaded file extension is allowed.
    """

    if not filename:
        return False

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def generate_random_file_name(filename: str) -> str:
    """
    Generate a unique filename while keeping its extension.
    """

    file_extension = Path(filename).suffix.lower()

    return f"{uuid.uuid4()}{file_extension}"


def get_pakistan_time() -> str:
    """
    Return current date and time in Pakistan timezone.
    """

    pakistan_timezone = pytz.timezone("Asia/Karachi")

    current_time = datetime.now(
        pakistan_timezone
    )

    return current_time.strftime("%Y-%m-%d %H:%M:%S")


# ==================================================
# 5. Fire Detection API
# ==================================================

@router.post(
    "/predict_fire",
    response_model=PredictFireSchema
)
async def predict_fire(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    logger.info("--------------------------------")
    logger.info(f"Received file: {file.filename}")

    # ----------------------------------------------
    # Step 1: Validate uploaded file
    # ----------------------------------------------

    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported file format. "
                "Only JPG, JPEG and PNG are allowed."
            )
        )

    # ----------------------------------------------
    # Step 2: Generate unique filename
    # ----------------------------------------------

    new_file_name = generate_random_file_name(
        file.filename
    )

    temp_file_path = TEMP_DIR / new_file_name

    # ----------------------------------------------
    # Step 3: Save temporary uploaded image
    # ----------------------------------------------

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(
            f"Temporary image saved: {temp_file_path}"
        )

    except OSError as e:
        logger.error(f"Failed to save image: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded image."
        )

    finally:
        await file.close()

    # ----------------------------------------------
    # Step 4: Run YOLO inference on CPU
    # ----------------------------------------------

    try:

        logger.info("Running fire detection on CPU...")

        results = model.predict(
            source=str(temp_file_path),
            device="cpu",
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes

        # Draw detected bounding boxes
        annotated_img = result.plot()

        # ------------------------------------------
        # Step 5: Process detection results
        # ------------------------------------------

        processed_result = {
            "file_name": file.filename,
            "detections": []
        }

        fire_detected = False

        for box in boxes:

            class_id = int(box.cls.item())

            class_name = model.names[class_id]

            confidence = float(box.conf.item())

            bbox = box.xyxy[0].tolist()

            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox
            )

            processed_result["detections"].append(
                detection
            )

            if class_name.lower() == "fire":
                fire_detected = True

        # ------------------------------------------
        # Step 6: Pakistan date and time
        # ------------------------------------------

        current_time = get_pakistan_time()

        # ------------------------------------------
        # Step 7: Save result image if fire detected
        # ------------------------------------------

        if fire_detected:

            log_file_path = LOG_DIR / new_file_name

            saved = cv2.imwrite(
                str(log_file_path),
                annotated_img
            )

            if not saved:
                raise RuntimeError(
                    "Failed to save annotated image."
                )

            logger.info(
                f"Result image saved: {log_file_path}"
            )

            response_result = {
                "message": "fire detected",
                "file_name": file.filename,
                "detections": processed_result["detections"],
                "result_image": new_file_name,
                "date": current_time
            }

        else:

            logger.info(
                "No fire detected in uploaded image."
            )

            response_result = {
                "message": "safe",
                "file_name": None,
                "detections": processed_result["detections"],
                "result_image": None,
                "date": current_time
            }

        # ------------------------------------------
        # Step 8: Save detection log to database
        # ------------------------------------------

        create_detection_log(
            db=db,
            detection_data=response_result
        )

        logger.info(
            f"Response result: {response_result}"
        )

        return response_result

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            f"Error processing image: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process image."
        )

    # ----------------------------------------------
    # Step 9: Delete temporary uploaded image
    # ----------------------------------------------

    finally:

        logger.info(
            "Cleaning up temporary files..."
        )

        try:

            if temp_file_path.exists():

                temp_file_path.unlink()

                logger.info(
                    "Temporary image deleted successfully."
                )

        except OSError as e:

            logger.warning(
                f"Failed to delete temporary file: {e}"
            )
