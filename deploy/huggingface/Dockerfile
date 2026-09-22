# Hugging Face Spaces (Docker) — daimi pulsuz hostinq üçün
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p logs && python -c "from inference.detector import VehicleDetector; VehicleDetector()._load()" \
 && python -c "from inference.zeroshot import ClipZeroShot; ClipZeroShot()._load()"
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
