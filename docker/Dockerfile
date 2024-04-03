FROM continuumio/miniconda3:4.6.14

RUN echo "deb http://archive.debian.org/debian stretch main" > /etc/apt/sources.list

# get data
RUN mkdir /data/
RUN apt-get update && apt-get install -y p7zip-full
RUN cd /data && wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O fma_small.zip && 7z x fma_small.zip && rm fma_small.zip

# install tensorflow
RUN conda install -c anaconda tensorflow==1.12

# install pandas
RUN conda install -y -c anaconda pandas

# install ffmpeg (for loading mp3 files)
RUN conda install -y ffmpeg

RUN pip install librosa

#RUN apt-get install libffi-dev
#RUN ln -s /usr/lib/x86_64-linux-gnu/libffi.so.6 /usr/lib/x86_64-linux-gnu/libffi.so.7

# clone git repo of the project
RUN mkdir /workspace/ && cd /workspace/ && git clone https://github.com/AntoninLngt/genre_classification.git
WORKDIR /workspace/genre_classification

ENTRYPOINT ["bash"]
