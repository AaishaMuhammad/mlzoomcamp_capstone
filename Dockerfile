FROM python:3.10-slim

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY docker_req.txt .
RUN pip install -r docker_req.txt

COPY kitchenware_model.onnx .
COPY predictions.py .
COPY service.py .

EXPOSE 8080
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080", "service:app"]

