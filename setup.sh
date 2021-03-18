# Run the first time on server
conda create --name m5
conda activate m5
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
python -m pip install detectron2==0.4 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
pip install opencv-python
pip install pandas
pip install matplotlib