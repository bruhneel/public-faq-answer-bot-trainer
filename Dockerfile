FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["sh", "-c", "streamlit run app/main.py --server.address 0.0.0.0 --server.port ${PORT}"]
