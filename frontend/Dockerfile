# Slim base: the frontend never touches CUDA.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py presets.py style.css ./
COPY examples ./examples
# Batch-tab CSV exports land here; writable by whichever UID compose runs us as.
RUN mkdir -p /app/outputs && chmod 1777 /app/outputs

EXPOSE 7860
CMD ["python", "app.py"]
