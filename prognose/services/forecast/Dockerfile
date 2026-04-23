FROM python:3.13-slim
WORKDIR /app
COPY services/forecast/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services/forecast /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
