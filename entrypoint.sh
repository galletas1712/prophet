cd /docker_build

# TODO: careful about .bashrc!
if [ ! -d /root/miniconda3 ]; then
	wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
	bash Miniforge3.sh -b -p "/root/miniconda3"
	rm Miniforge3.sh

	export PATH="/root/miniconda3/bin:$PATH"

	# Create environment
	mamba init bash && . /root/.bashrc && mamba create -n prophet python=3.12 -y
	mamba activate prophet
	pip install -r requirements.txt

	echo "mamba activate prophet" >>/root/.bashrc
	echo "export PYTHONPATH=/root/prophet" >>/root/.bashrc
fi

cd /root
rm -r /docker_build

mkdir -p /root/.ssh
echo $AUTHORIZED_KEYS >/root/.ssh/authorized_keys
/usr/sbin/sshd

sleep inf
