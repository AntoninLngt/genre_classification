FROM continuumio/miniconda3:4.6.14

RUN echo "deb http://archive.debian.org/debian stretch main" > /etc/apt/sources.list

# get data
RUN mkdir /data/
RUN apt-get update && apt-get install -y p7zip-full
RUN cd /data && wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O fma_small.zip && 7z x fma_small.zip && rm fma_small.zip

# install tensorflow
RUN conda install -c anaconda tensorflow==2.14

# install pandas
RUN conda install -y -c anaconda pandas

# install ffmpeg (for loading mp3 files)
RUN conda install -y ffmpeg

# clone git repo of the project
RUN mkdir /workspace/ && cd /workspace/ && git clone https://github.com/AntoninLngt/genre_classification.git
WORKDIR /workspace/genre_classification

ENTRYPOINT ["bash"]
