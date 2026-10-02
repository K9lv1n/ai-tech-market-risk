FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py schemas.py model_service.py ./
COPY artifacts/ ./artifacts/

# Fail the image build early when the trained inference artifact is absent.
RUN test -f artifacts/logistic_regression_candidate.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
