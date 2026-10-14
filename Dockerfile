FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gdal-bin libgdal-dev build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/frontend/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
