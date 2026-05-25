FROM python:3.10-slim-bookworm

WORKDIR /opt/api

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.31.0

COPY services/api/ ./

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
