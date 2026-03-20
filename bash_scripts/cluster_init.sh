# enroot create -n clare /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/enroot-images/nvidia+cuda+12.6.3-cudnn-devel-ubuntu22.04.sqsh

# enroot start \
#     --root --rw \
#     --mount /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/yi_zhang/lerobot:/workspace/lerobot \
#     --mount /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/yi_zhang/peft:/workspace/peft \
#     --mount /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/huggingface:/workspace/huggingface \
#     --mount /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/checkpoints/clare:/workspace/lerobot/outputs \
#     clare

apt update
apt install -y ffmpeg libevdev-dev git software-properties-common
apt-get install git-lfs
git lfs install
add-apt-repository -y ppa:deadsnakes/ppa
apt update 
apt install -y python3.12 python3.12-venv python3.12-dev
update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
update-alternatives --set python /usr/bin/python3.12
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
update-alternatives --set python3 /usr/bin/python3.12
python -m ensurepip
python -m pip install --upgrade pip setuptools
ln -sf /usr/local/bin/pip3 /usr/local/bin/pip
ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip 2>/dev/null || true
cd /workspace/lerobot/ 
pip install -e .
cd /workspace/peft
pip install -e .
pip install deepspeed

pip cache remove evdev 2>/dev/null || true
apt install vim

# enroot export -o ./clare.sqsh -f clare
