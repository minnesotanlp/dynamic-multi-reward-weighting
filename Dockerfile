FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# fix for nvidia key issue
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
    vim \
    sudo

# add non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
ENV HOME=/home/user
RUN chmod 777 /home/user

# install project requirements
RUN pip install accelerate datasets transformers trl pandas peft protobuf==3.20.* wandb

