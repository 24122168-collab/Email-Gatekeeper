#Base image
FROM python:3.11-slim

#Set work directory
WORKDIR /app

#Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy project files
COPY . .

#Expose the port Hugging Face expects
EXPOSE 7860

#Command to run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]