#Dockerfile
# FROM continuumio/anaconda3
FROM continuumio/miniconda3 

RUN  chmod -R 777 /usr/local/*
#RUN apt-cache madison python3-opencv
RUN apt update -qq && apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 build-essential vim curl wget libhdf5-dev libhdf5-serial-dev cython3 tesseract-ocr


ARG UNAME=docker
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
# RUN chown -R 1000:1000 /opt/conda
USER $UNAME
WORKDIR /home/docker 
# COPY --chown=docker:docker *.ipynb /home/docker/ 
# COPY --chown=docker:docker *.sh /home/docker/  
COPY --chown=docker:docker *.yml /home/docker/
COPY --chown=docker:docker .  /home/docker/

RUN chmod +x ./entrypoint.sh

RUN conda env create --file /home/docker/yolo8.yml -n yolo8
RUN echo "source activate yolo8" > ~/.bashrc
ENV PATH /opt/conda/envs/yolo8/bin:$PATH

RUN chmod +x ~/.bashrc
RUN chown docker:docker ~/.bashrc

ENTRYPOINT ["./entrypoint.sh"]