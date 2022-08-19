cd /WORKDIR

gh auth login

git clone https://github.com/Pappol/simswap_on_edge

cd simswap_on_edge 
conda env create -f environment.yml

conda activate simswap 

mkdir ./arcface_model

pip install gdown

cd ./arcface_model

gdown https://drive.google.com/u/0/uc?id=1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N&export=download

cd .. 
python python train.py --name v100 --batchSize 16
