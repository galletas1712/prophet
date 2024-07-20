FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

SHELL ["/bin/bash", "-c"]
WORKDIR "/root"

RUN apt-get update && \
    apt-get install -y wget curl sudo git build-essential vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3.sh -b -p "/root/miniconda3"
RUN rm Miniforge3.sh

ENV PATH="/root/miniconda3/bin:$PATH"
RUN mamba init bash \
	&& . /root/.bashrc \
	&& mamba create -n prophet python=3.12 -y

COPY requirements.txt /root/requirements.txt
RUN . /root/miniconda3/etc/profile.d/conda.sh && . /root/miniconda3/etc/profile.d/mamba.sh && mamba activate prophet && pip install -r requirements.txt
RUN rm /root/requirements.txt

RUN echo "mamba activate prophet" >> /root/.bashrc
RUN echo "export PYTHONPATH=/root/prophet" >> /root/.bashrc
