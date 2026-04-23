FROM python:3.13-slim
WORKDIR /app
COPY scripts/demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY scripts/demo/generate_counts.py /app/generate_counts.py
CMD ["python", "/app/generate_counts.py"]
