FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]