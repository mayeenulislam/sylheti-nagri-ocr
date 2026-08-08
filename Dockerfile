# syntax=docker/dockerfile:1
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgfortran5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py streamlit_vit.py configs.yaml vit_configs.yaml ./
COPY Surma-4.000/Surma-Regular.ttf Surma-4.000/
COPY model_weights.weights.h5 vit_model_weights.weights.h5 ./

USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
