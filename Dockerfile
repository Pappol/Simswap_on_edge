FROM nvidia/cuda:11.7.0-base-ubuntu22.04

RUN apt update

RUN apt-get install -y python3 python3-pip

RUN pip install tensorflow-gpu

RUN  install git && cd /home && git clone https://github.com/Pappol/simswap_on_edge

RUN  cd simswap_on_edge && git pull && \
	apt update && mkdir /home/simswap_on_edge/arcface_model && pip install gdown &&\
    cd /home/simswap_on_edge/arcface_model && gdown https://drive.google.com/u/0/uc?id=1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N&export=download && \
    conda env create -f environment.yml && conda activate simswap cd .. && python python train.py --name v100 --batchSize 16

RUN  conda activate simswap && cd ./arcface_model &&  gdown https://drive.google.com/u/0/uc?id=1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N&export=download && \
    cd /home/simswap_on_edge/&& python python train.py --name v100 --batchSize 16

WORKDIR /home/simswap_on_edge
