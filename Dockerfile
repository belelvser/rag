# Один официальный Python 3.11 slim, зафиксированный по digest (Python 3.11.17).
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS api

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 OTEL_SDK_DISABLED=true
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-api.txt \
    && python -m pip check \
    && python -c "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('sentence_transformers') is None"
COPY app ./app
COPY scripts/check_docker.py ./scripts/check_docker.py
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM api AS agent
COPY requirements-agent.txt ./
# Только Linux binary wheels. CPU-index исключает установку CUDA-сборки torch.
RUN python -m pip install --no-cache-dir --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu \
    && python -m pip install --no-cache-dir --only-binary=:all: -r requirements-agent.txt \
    && python -m pip check \
    && python -c "import torch,importlib.metadata as m; assert torch.version.cuda is None; assert not any(d.metadata['Name'].lower().startswith('nvidia-') for d in m.distributions()); print('Linux CPU torch:',torch.__version__)"
COPY agent ./agent
COPY scripts/retrieval.py scripts/rag.py scripts/index_qdrant.py scripts/init_services.py ./scripts/
CMD ["python", "-m", "uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "9999", "--workers", "1"]
