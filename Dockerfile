FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 7860 & python inference.py && tail -f /dev/null"]
