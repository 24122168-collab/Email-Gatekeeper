# Base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# --- YAHAN CHANGE HAI ---
# Command to run Gradio app directly
CMD ["python", "app.py"]