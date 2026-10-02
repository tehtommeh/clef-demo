# torch 2.11 is what the model card was tested with. CUDA 12.8 runs on any
# driver >= 570; override API_BASE_IMAGE for other hosts.
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HOME=/tmp

# Triton JIT-compiles a small C launcher at runtime, so it needs a C compiler.
# The 2.11 image ships gcc; install one only if a different base lacks it.
RUN command -v cc >/dev/null || (apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*)

WORKDIR /app
COPY requirements.txt .
# The base image's Python is the system one (PEP 668); inside a container that
# is the intended interpreter, hence PIP_BREAK_SYSTEM_PACKAGES above.
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch, transformers, fla, av; print('torch', torch.__version__, 'transformers', transformers.__version__)"

COPY app.py .

EXPOSE 8000
# One worker: each worker would load its own 19GB copy of the weights.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
