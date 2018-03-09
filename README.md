# DHP

This repository provides [**database**](#download-and-setup-pvs-hm-database), [**code**](#setup-an-environment-to-run-our-code) and [**results visualization**](#results-visualization) for reproducing all the reported results in the paper:

* [**Modeling Attention in Panoramic Video: A Deep Reinforcement Learning Approach.**](https://arxiv.org/abs/1710.10755)
[*Yuhang Song* &#8224;](https://yuhangsong.my.cam/),
[*Mai Xu* &#8224;&#8727;](http://shi.buaa.edu.cn/MaiXu/),
[*Jianyi Wang*](http://45.77.201.133/html/Members/jianyiwang.html).
Submitted to [TPAMI](https://www.computer.org/web/tpami).
By [MC2 Lab](http://45.77.201.133/) @ [Beihang University](http://ev.buaa.edu.cn/).

<p align="center"><img src="https://github.com/YuhangSong/DHP/blob/master/imgs/VRBasketball_all.gif"/></p>

Specifically, this repository includes extremely simple guidelines to:
* [Download and setup the PVS-HMEM database](#download-and-setup-pvs-hm-database).
* [Setup a friendly environment to run our code.](#setup-an-environment-to-run-our-code)
* [Reproduce visualized results from the paper.](#results-visualization)

## Download and setup PVS-HMEM database

Our PVS-HMEM (Panoramic Video Sequences with Head Movement & Eye Movement database) database contains both **Head Movement** and **Eye Movement** data of **58** subjects on **76** panoramic videos.
* *Blue dots* represent the **Head Movement**.
* *Translucent blue circles* represent the **FoV**.
* *Red dots* represent the **Eye Movement**.

![](https://github.com/YuhangSong/DHP/blob/master/imgs/Snowfield_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/Catwalks_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/A380_all.gif)
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/YuhangSong/DHP/blob/master/imgs/SpaceWar2_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/Pearl_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/Predator_all.gif)
![](https://github.com/YuhangSong/DHP/blob/master/imgs/Camping_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/CandyCarnival_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/NotBeAloneTonight_all.gif)


Follow command lines here to download and setup our PVS-HM database:
```
mkdir -p dhp_env/
cd dhp_env/
wget https://drive.google.com/open?id=0B20VnLepDOl4aFhsT0x6YjA
tar -xzvf dataset.tar.gz
```

## Setup an environment to run our code

### Pre-requirements

* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [Anaconda3](https://www.anaconda.com/download/) (Python 3.6)

If you are not familiar with above things, refer to [my personal basic setup](https://github.com/YuhangSong/Cool-Ubuntu-For-DL) for some guidelines.
The code should also be runnable without a GPU, but I would not recommend it.

### Requirements

There will be command lines after the list, you don't have to install below requirements one by one.
Besides, if you are not familiar with below things, I highly recommend you to just follow command lines after the list:
* Python 3.6
* [Pytorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* [numpy](http://www.numpy.org/)
* [gym](https://github.com/openai/gym)
* [imageio](https://imageio.github.io/)
* [matplotlib](https://matplotlib.org/)
* [pybullet](https://pypi.python.org/pypi/pybullet)
* [opencv-python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

Install above requirements with command lines:
```
# create env
conda create -n grl_env

# source in env
source ~/.bashrc
source activate dhp_env

# install requirements
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl # if you are using CUDA 8.0, otherwise, refer to their official site: http://pytorch.org/
pip install torchvision
pip install visdom
pip install numpy -I
pip install gym[atari]
pip install imageio
pip install matplotlib
pip install pybullet
pip install opencv-python

# clean dir and create dir
mkdir -p dhp_env/project/
cd dhp_env/project/
git clone https://github.com/YuhangSong/DHP.git
cd DHP
```

Meet some issues? See [problems](https://github.com/YuhangSong/GTN#problems). If there isn't a solution, please don not hesitate to open an issue.

## Run our code

#### Start a `Visdom` server with
```bash
source ~/.bashrc
source activate dhp_env
python -m visdom.server
```
Visdom will serve `http://localhost:8097/` by default.

#### Run DHP.
```bash
source ~/.bashrc
source activate dhp_env
CUDA_VISIBLE_DEVICES=0 python main.py
```

#### Run other baselines.
Give ```arguments.py``` a look, it is well commented.

## Results Visualization

### Reward Function

We propose a reward function that can capture transition of the attention.

Our reward function            |  Baseline reward function       
:-------------------------:|:-------------------------:
<img src="imgs/our_transition.gif">  |  <img src="imgs/baseline_transition.gif">

Specifically, in above example, the woman and the man are passing the basketball between each other, and subjects' attention are switching between them while they passing the basketball.
Our reward function is able to capture these transitions of the attentions smoothly, while the baseline reward function makes the agent focus on the man all the time, even when the basketball is not in his hands.

## Authors
Yuhang Song            |  Mai Xu          |  Jianyi Wang
:-------------------------:|:-------------------------:|:-------------------------:
<img src="imgs/YuhangSong.png" width="200">  |  <img src="imgs/MaiXu.png" width="200">  |  <img src="imgs/JianyiWang.png" width="200">
