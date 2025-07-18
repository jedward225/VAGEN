wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
sudo apt-get update -y && sudo apt-get install -y vulkan-tools mesa-vulkan-drivers

pip install ai2thor
sudo apt-get update
sudo apt-get install -y vulkan-tools libvulkan1 vulkan-utils
sudo apt-get install -y xvfb mesa-utils
sudo apt install net-tools
sudo apt-get install -y libvulkan1
sudo apt install x11-apps
