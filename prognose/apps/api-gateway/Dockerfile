FROM python:3.13-slim
WORKDIR /app
COPY apps/api-gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY apps/api-gateway /app
COPY packages/shared/prompts/explainability /app/packages/shared/prompts/explainability
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
