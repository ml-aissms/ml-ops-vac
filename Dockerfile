FROM python:3.9-slim 
LABEL maintainer="MLOPS-VAC"
LABEL version="1.0"

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY ./requirements.txt /requirements.txt
COPY ./models/model.joblib /models/model.joblib
COPY ./webapp /app/webapp

EXPOSE 8000

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /requirements.txt && \
    rm -rf /root/.cache && \
    adduser --disabled-password --no-create-home appuser

ENV PATH="/py/bin:$PATH" 

USER appuser

