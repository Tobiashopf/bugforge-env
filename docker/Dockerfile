# Sandbox image for running agent code and grading:
#   docker build -t bugforge-sandbox docker
FROM python:3.12-slim
RUN pip install --no-cache-dir "pytest>=7,<10" && useradd --create-home runner
USER runner
WORKDIR /work
