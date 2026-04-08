FROM python:3.11-slim

# User setup for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy and install as user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Expose port
EXPOSE 7860

# --- YAHAN CHANGE HAI ---
# Direct uvicorn ki jagah hum python module run karenge jo main() ko call karega
CMD ["python", "-m", "server.app"]
