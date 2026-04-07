from python:3.10-slim
workdir /app
copy requirements.txt
run pip install --no-cache-dir -r requirements.txt
copy . .
expose 8501
cmd ["streamlit","run","src/app.py","--sever.port=8501","--server.address=0.0.0.0"]