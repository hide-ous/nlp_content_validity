# Use multi-stage for frontend build
FROM node:18 AS frontend-build
WORKDIR /app
COPY frontend/ /app
RUN npm install && npm run build

# Backend with uvicorn + FastAPI
FROM python:3.11-slim
WORKDIR /app

# Backend setup
COPY ../../data/ /app/data/
COPY backend/ /app/backend/
#COPY --from=frontend-build /app/dist /app/frontend-dist
COPY --from=frontend-build /app/dist /frontend/dist
#COPY frontend/dist /app/frontend-dist


# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API entry point
COPY start.py .

EXPOSE 8002
CMD ["python", "start.py"]
