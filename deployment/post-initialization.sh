#!/bin/bash

cd ${HOME}
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
rm Miniforge3.sh
echo 'source "${HOME}/conda/etc/profile.d/conda.sh"' >${HOME}/.bashrc
echo 'source "${HOME}/conda/etc/profile.d/mamba.sh"' >>${HOME}/.bashrc

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
rm vscode_cli.tar.gz

sleep inf
