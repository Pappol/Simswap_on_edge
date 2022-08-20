FROM continuumio/anaconda3


RUN apt-get update -y && apt-get upgrade -y && apt-get install apt-utils


RUN  apt-get install git -y
SHELL ["/bin/bash", "-c"]
RUN	conda create -n simswap python=3.9 && source ~/.bashrc && conda activate simswap && \
     conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y && \
	pip install --ignore-installed imageio && \
	pip install insightface==0.2.1 onnxruntime moviepy && \
     conda install -c conda-forge timm -y 

WORKDIR /home/
