FROM tensorflow/tensorflow:latest-gpu AS GPU-KERAS
COPY . /code
WORKDIR /code
RUN pip3 install -r /code/requirements.txt
ENV TEST_VAR="this is a test"
LABEL "name"="gpu-keras"