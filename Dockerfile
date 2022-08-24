# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.05-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# Install linux packages
RUN apt update 

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
RUN pip install --no-cache -r requirements.txt

# Create working directory
RUN mkdir -p /home/workspace
WORKDIR /home/workspace
