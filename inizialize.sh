mkdir ./arcface_model

pip install gdown

cd ./arcface_model

gdown https://drive.google.com/u/0/uc?id=1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N&export=download

cd .. 
gdown https://drive.google.com/u/0/uc?id=19pWvdEHS-CEG6tW3PdxdtZ5QEymVjImc&export=download

tar -xvf vggface2_crop_arcfacealign_224.tar

python train.py --name v100 --batchSize 4
