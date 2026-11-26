FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN useradd --create-home --uid 10001 app

COPY requirements.txt docker-constraints.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --constraint docker-constraints.txt \
        -r requirements.txt

COPY --chown=app:app . .
RUN mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

USER app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
