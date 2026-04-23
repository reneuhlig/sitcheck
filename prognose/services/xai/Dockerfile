FROM python:3.13-slim
WORKDIR /app
COPY services/xai/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services/xai /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
