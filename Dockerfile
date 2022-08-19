FROM continuumio/anaconda3

RUN  apt install git && cd /home && git clone https://github.com/Pappol/simswap_on_edge && cd simswap_on_edge && git pull && \
	apt update && apt upgrade -y && 

RUN	cd /home/simswap_on_edge && conda env create -f environment.yml

RUN  conda activate simswap && /home/simswap_on_edge/arcface_model &&  gdown https://drive.google.com/u/0/uc?id=1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N&export=download && \
    cd /home/simswap_on_edge/ && python python train.py --name v100 --batchSize 16

WORKDIR /home/simswap_on_edge
