FROM python:3.12-slim

WORKDIR /app

COPY services/lecture-ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/lecture-ingest/main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
