FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 as base

SHELL ["/bin/bash", "-c"]

# Install core stuff
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gnupg && \
    apt-get install -y wget curl sudo git build-essential openssh-server

# Install nsight and nvidia container toolkit
RUN apt-get install -y --no-install-recommends gnupg && \
	echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
	apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
	apt-get update && \
	apt-get install nsight-systems-cli -y

RUN apt-get update && apt-get install -y nvidia-cuda-toolkit

# Install vanity stuff
RUN apt-get install -y gh neovim htop && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN cp /etc/ssh/sshd_config /etc/ssh/sshd_config-original \
 && sed -i 's/^#\s*Port.*/Port 2222/' /etc/ssh/sshd_config \
 && sed -i 's/^#\s*PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && mkdir -p /root/.ssh \
 && chmod 700 /root/.ssh \
 && mkdir /var/run/sshd \
 && chmod 755 /var/run/sshd \
 && rm -rf /var/lib/apt/lists /var/cache/apt/archives

WORKDIR /root

RUN mkdir /docker_build
COPY --chmod=0777 ./entrypoint.sh /docker_build/entrypoint.sh
COPY --chmod=0777 ./requirements.txt /docker_build/requirements.txt

CMD ["/bin/bash", "-c", "/docker_build/entrypoint.sh"]
