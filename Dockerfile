FROM python:3.12-slim

WORKDIR /app

COPY docker_requirements .
RUN pip install --no-cache-dir -r docker_requirements

COPY app/ ./app

CMD ["python", "app/main.py"]