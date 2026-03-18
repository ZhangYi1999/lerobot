sudo apt update
sudo apt install screen -y


cd /workspace/lerobot
pip install nvitop
pip install -e .
pip install transformers==4.57.1
pip install torchcodec==0.6