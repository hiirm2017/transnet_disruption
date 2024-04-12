FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN apt-get update && \
    apt-get install -y libgdal-dev && \
    apt-get install -y g++ && \
    pip install --upgrade pip && \
    pip install poetry && \
    poetry install --no-root

ENV PYTHONPATH /app
