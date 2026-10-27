# FlameGuard

Real-time fire detection from a webcam using a custom YOLO model, FastAPI, SQLite, and Next.js.

FlameGuard captures frames from a selected camera, sends them to a local inference API, and displays the result in the browser. When fire is detected, the app highlights the event, plays an alarm tone, stores the detection metadata, and makes the annotated image available in the detection history.

> **Safety notice:** FlameGuard is a software demonstration and decision-support tool. It is not a certified fire alarm or life-safety system. Do not rely on it as the only method of detecting fire or protecting people and property.

## Features

- Webcam selection through the browser's available video devices
- Continuous frame capture and inference polling while the stream is running
- YOLO-based fire detection with bounding boxes and confidence scores
- Browser alarm sound, visual flash, and notification when fire is detected
- SQLite-backed detection history with pagination
- Annotated images served by the FastAPI backend
- Separate frontend and backend development workflows
- Reproducible Python dependencies in [`backend/environment.yml`](./backend/environment.yml)

## Architecture

```text
Browser webcam
     |
     | JPEG frame every 5 seconds
     v
Next.js frontend (:3000)
     |
     | POST /predict_fire
     v
FastAPI backend (:8000)
     |
     +--> YOLO model (CPU inference)
     +--> SQLite detection log
     +--> Annotated positive images in backend/app/log/
```

### Project layout

```text
FlameGuard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── create_user/
│   │   │   ├── get_detection_log/
│   │   │   ├── get_test/
│   │   │   └── predict_fire/
│   │   ├── assets/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── db/
│   │   ├── log/                 # generated fire-positive images
│   │   └── main.py
│   └── environment.yml
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx             # live camera detection
│   │   └── detectionLogs/       # detection history
│   ├── package.json
│   └── pnpm-lock.yaml
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.9 (the provided Conda environment is pinned to 3.9.21)
- Conda or Miniconda
- Node.js 20+ and pnpm
- A webcam for live detection

The included model weights are designed for CPU inference. A CUDA-enabled PyTorch setup can be used separately if GPU inference is required.

## Quick start

Clone the repository and enter the project directory:

```bash
git clone https://github.com/abdullahjave/Fire-Safty-Detection.git
cd Fire-Safty-Detection
```

### 1. Start the backend

Create and activate the Conda environment:

```bash
cd backend
conda env create -f environment.yml
conda activate flameguard
```

Start FastAPI from the backend directory:

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at <http://localhost:8000>. FastAPI's interactive documentation is available at <http://localhost:8000/docs>.

### 2. Start the frontend

Open another terminal at the project root:

```bash
cd frontend
pnpm install
pnpm dev
```

Open <http://localhost:3000>, allow camera and audio access, select a camera, and choose **Start Stream**.

<img width="332" height="438" alt="Screenshot 2026-10-27 103459" src="https://github.com/user-attachments/assets/99c9e4c8-a01d-4d1f-bb12-be0f77c19eff" />

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/predict_fire` | Analyze an uploaded JPG, JPEG, or PNG image |
| `GET` | `/get_detection_log?page=1&page_size=10` | Return paginated detection history |
| `POST` | `/create_user/` | Create a user record |
| `GET` | `/get_test` | Basic API health/test response |
| `GET` | `/get_test2` | Basic API health/test response |
| `GET` | `/log/{filename}` | Serve an annotated detection image |

### Example inference request

```bash
curl -X POST http://localhost:8000/predict_fire ^
  -F "file=@path\\to\\image.jpg"
```

The response contains the status (`fire detected` or `safe`), detection class names, confidence values, bounding boxes, timestamp, and the annotated result image name when fire is detected.

## Model and training notes

The application loads `backend/app/assets/best.pt` at startup. The repository also contains `last.pt` as a training checkpoint. The model can be retrained with a compatible fire-detection dataset, then the resulting `best.pt` can replace the bundled weights.

Typical workflow:

1. Prepare and label a fire-detection dataset.
2. Train a YOLO model using the Ultralytics tooling.
3. Copy the resulting `best.pt` into `backend/app/assets/`.
4. Restart the backend so the model is loaded.
5. Validate predictions on representative positive and negative images.

## Data and privacy

- Uploaded frames are written to a temporary backend directory during inference.
- Temporary files are removed after processing.
- Annotated images are saved only for fire-positive results and are exposed through `/log/`.
- The SQLite database and generated logs are local runtime data and are excluded from version control.
- Camera access is requested by the browser and requires user permission.

## Development

Useful commands:

```bash
# frontend
cd frontend
pnpm dev
pnpm build

# backend
cd backend/app
uvicorn main:app --reload
```

The frontend currently targets `http://localhost:8000` for the backend API. If the services are hosted elsewhere, update the API URLs in the frontend pages before deployment and configure CORS appropriately in the backend.

<img width="353" height="634" alt="m" src="https://github.com/user-attachments/assets/fc418742-40a2-4756-8c63-15ffc4f7f047" />

## Known limitations

- The frontend currently uses a fixed local backend URL.
- Inference is configured for CPU and may be slow on low-powered machines.
- Detection quality depends on the training data, camera quality, lighting, and the selected confidence threshold.
- CORS is permissive for local development and should be restricted before production deployment.
- This project does not replace a certified fire-safety system.

## License

No license has been selected for this repository yet. Add a license before distributing or reusing the project.
