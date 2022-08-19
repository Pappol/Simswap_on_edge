FROM nvidia/cuda:11.7.0-base-ubuntu22.04

RUN apt update && apt-get upgrade -y && apt-get install apt-utils

# Install base utilities
RUN apt-get install build-essentials -y && \
    apt-get install wget -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH



RUN apt-get install -y python3 python3-pip

RUN pip install tensorflow-gpu

RUN  apt-get install git -y && cd /home && apt-get install gh

WORKDIR /home/simswap_on_edge
