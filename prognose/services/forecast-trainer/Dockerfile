FROM python:3.13-slim
WORKDIR /app
COPY services/forecast-trainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services/forecast-trainer/main.py /app/main.py
RUN mkdir -p /app/runtime
CMD ["python", "main.py"]
