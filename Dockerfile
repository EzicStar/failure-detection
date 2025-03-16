FROM python:3.8-slim-buster
WORKDIR /python-docker
EXPOSE 3333

# Dependencies required for open-cv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 wget -y
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN echo 'Downloading the latest failure detection AI model in ONNX format...'
RUN wget -O model/model-weights.onnx $(cat model/model-weights.onnx.url | tr -d '\r')

ENTRYPOINT [ "python" ]
CMD [ "server.py"]
