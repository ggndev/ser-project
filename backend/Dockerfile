FROM python:3.11-slim
# libsndfile for soundfile (decoding); curl to fetch the model from the GitHub Release at build time
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Model file is too big for the GitHub website upload, so it lives in a Release asset (change URL if you re-release)
ARG MODEL_URL=https://github.com/ggndev/ser-project/releases/download/v1/model.onnx
RUN if [ ! -f model/model.onnx ]; then curl -fL --retry 3 -o model/model.onnx "$MODEL_URL"; fi \
    && ls -la model
ENV PORT=10000 ORT_THREADS=1
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
