FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu

RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    Werkzeug==3.0.1 \
    joblib==1.3.2 \
    pandas==2.1.4 \
    scikit-learn==1.8.0 \
    numpy==1.26.3 \
    opencv-python-headless==4.9.0.80 \
    ultralytics==8.1.0

COPY app.py /app/app.py
COPY docker_smoke_test.py /app/docker_smoke_test.py
COPY templates /app/templates
COPY GrievX_UI_and_model/artifacts/grievx_resolution_model.joblib /app/GrievX_UI_and_model/artifacts/grievx_resolution_model.joblib
COPY GrievX_UI_and_model/artifacts/grievx_metadata.json /app/GrievX_UI_and_model/artifacts/grievx_metadata.json
COPY GrievX_UI_and_model/Final_MLA_and_Ward_member_Dataset.csv /app/GrievX_UI_and_model/Final_MLA_and_Ward_member_Dataset.csv
COPY runs/grievx_yolov8s_cpu/weights/best.pt /app/runs/grievx_yolov8s_cpu/weights/best.pt

RUN python docker_smoke_test.py --mode build

EXPOSE 5000

CMD ["python", "app.py"]