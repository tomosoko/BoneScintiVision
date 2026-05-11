FROM python:3.12-slim

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights must be provided at build or runtime:
#   docker build --build-arg MODEL_WEIGHTS=runs/detect/.../best.pt .
#   or: docker run -v /host/weights:/app/runs/detect/bone_scinti_detector_v8/weights ...
EXPOSE 8765

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8765"]
