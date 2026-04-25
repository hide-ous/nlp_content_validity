# Use multi-stage for frontend build
FROM node:18 AS frontend-build
WORKDIR /app
COPY nlp_content_validity/visualization/frontend/ /app/
RUN npm install && npm run build

# Backend with uvicorn + FastAPI
FROM python:3.11-slim
WORKDIR /app

ENV MODEL_CACHE_DIR=/app/models
ENV GENSIM_DATA_DIR=/app/models/gensim
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers

# Backend setup
COPY data/ /app/data/
COPY nlp_content_validity/visualization/backend/ /app/backend/
COPY --from=frontend-build /app/dist /app/frontend/dist

# Install dependencies
COPY nlp_content_validity/visualization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API entry point
COPY nlp_content_validity/visualization/start.py .

RUN mkdir -p /app/models/gensim /app/models/sentence_transformers
VOLUME ["/app/models"]

EXPOSE 8002
CMD ["python", "start.py"]