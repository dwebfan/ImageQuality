FROM pytorch/pytorch

ENV TZ="America/Los_Angeles"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt-get install -y python3-opencv
RUN pip install --upgrade pip && pip install opencv-python
RUN pip install piq

# run samples comparison to download model
RUN mkdir -p /piq/samples
COPY image_metrics.py /piq/image_metrics.py
COPY samples/20030117_1_320_0.ppm /piq/samples/
COPY samples/20030117_1_320_0.jpg /piq/samples/
COPY samples/20030117_1_320_0.webp /piq/samples/

RUN cd /piq && ./image_metrics.py samples/20030117_1_320_0.ppm samples/20030117_1_320_0.jpg samples/20030117_1_320_0.webp
