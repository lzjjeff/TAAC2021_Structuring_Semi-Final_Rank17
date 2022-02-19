#!/usr/bin/env bash

conda env create -f environment.yml
source activate
conda activate taac2021
python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
pip install paddleocr
pip install ffmpeg-python
pip install pytest-runner
pip install pytest
pip install addict
pip install pyyaml
pip install yapf
pip install opencv-python
pip install mmcv
pip install transformers
pip install prefetch_generator
sudo ln -s /opt/conda/envs/taac2021/lib/libcudnn.so.7.6.5 /usr/lib/libcudnn.so
sudo ln -s /opt/conda/envs/taac2021/lib/libcublas.so.10.2.1.243 /usr/lib/libcublas.so
sudo ln -s /opt/conda/envs/taac2021/lib/libcusolver.so.10.2.0.243 /usr/lib/libcusolver.so

mkdir save pretrain_models
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
cd pretrain_models
git clone https://huggingface.co/google/vit-base-patch16-224-in21k
git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext
cd ../

echo "Copy dataset ..."
time cp -r -d /home/tione/notebook/algo-2021/dataset /home/tione/notebook/
echo "Finished."