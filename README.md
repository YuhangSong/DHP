# DHP

This repository provides [**database**](#download-and-setup-pvs-hm-database), [**code**](#setup-an-environment-to-run-our-code) and [**results visualization**](#results-visualization) for reproducing all the reported results in the paper:

* [**Modeling Attention in Panoramic Video: A Deep Reinforcement Learning Approach.**](https://arxiv.org/abs/1710.10755)
[*Yuhang Song* &#8224;](https://yuhangsong.my.cam/),
[*Mai Xu* &#8224;&#8727;](http://shi.buaa.edu.cn/MaiXu/),
[*Jianyi Wang*](http://45.77.201.133/html/Members/jianyiwang.html).
Submitted to [TPAMI](https://www.computer.org/web/tpami).
By [MC2 Lab](http://45.77.201.133/) @ [Beihang University](http://ev.buaa.edu.cn/).

<p align="center"><img src="https://github.com/YuhangSong/DHP/blob/master/imgs/VRBasketball_all.gif"/></p>

Specifically, this repository includes guidelines to:
* [Download and setup the PVS-HMEM database](https://www.dropbox.com/s/ml03dhphyhj0j6u/PVS-HMEM%20database.tar.gz?dl=0). (Please contact us for the password)
* [Setup a environment to run our code.](#setup-an-environment-to-run-our-code)
* [Reproduce visualized results from the paper.](#results-visualization)

**Warning**: We have been working a updated version of DHP based on PyTorch with much more friendly setup procedures and strong GPU acceleration (The structure of the code is also cleaner).
This project is currently maintained, but will be depreciated in the future.

See [DHP-PyTorch](https://github.com/YuhangSong/DHP-pytorch) for the updated version. (Currently unavailable due to the copyright of our work)

## Download and setup PVS-HMEM database

Our PVS-HMEM (Panoramic Video Sequences with Head Movement & Eye Movement database) database contains both **Head Movement** and **Eye Movement** data of **58** subjects on **76** panoramic videos.
* *Blue dots* represent the **Head Movement**.
* *Translucent blue circles* represent the **FoV**.
* *Red dots* represent the **Eye Movement**.

![](https://github.com/YuhangSong/DHP/blob/master/imgs/Snowfield_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/Catwalks_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/A380_all.gif)
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/YuhangSong/DHP/blob/master/imgs/SpaceWar2_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/Pearl_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/Predator_all.gif)
![](https://github.com/YuhangSong/DHP/blob/master/imgs/Camping_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/CandyCarnival_all.gif)  |  ![](https://github.com/YuhangSong/DHP/blob/master/imgs/NotBeAloneTonight_all.gif)

Download our PVS-HM database from [DropBox link](xx), then extract it with:
```
tar -xzvf PVS-HM.tar.gz
```
Note that it contains all MP4 files of our database, along with the HM & EM scanpath data ```FULLdata_per_video_frame.mat```.

## Setup an environment to run our code

If you are not familiar with things in this section, refer to [my personal basic setup](https://github.com/YuhangSong/Cool-Ubuntu-For-DL) for some guidelines or simply google it.

Install [Anaconda](https://www.anaconda.com/) according to the guidelines on their [official site](https://www.anaconda.com/download/), then install other requirements with command lines:
```
source ~/.bashrc

# create env
conda create -n dhp_env python=2.7

# active env
source activate dhp_env

# install packages
pip install gym tensorflow universe

# clone project
git clone https://github.com/YuhangSong/DHP-TensorFlow.git

# make remap excuatble
cd DHP-TensorFlow
chmod +x ./remap
# you may run ./remap here to make sure the remap is excuatble
```

## Run our code

Please make sure you have:
* More than 64 GB of RAM.
* More then 600 GB space on the disk you store PVS-HM database.

#### Offline-DHP.

This section clarifies procedures to train and test offline-DHP.

##### Train

Set the ```database_path``` in ```config.py``` to your database folder.

Before trainning offline-DHP, generate YUV files. Set ```mode = 'data_processor'``` and ```data_processor_id = 'mp4_to_yuv'``` in ```config.py``` and run:
```bash
source ~/.bashrc
source activate dhp_env
python train.py
```
The converted YUV files will take about 600 Gb.
The reason we have to use YUV files is that, the remap function that get FoV from a 360 image is a binary file that takes YUV and output YUV.
We have developed a Python version of remap, but it turns out to be even slower than just reading and writing YUV files into the disk (for more then 5 times).
We are trying to see if remap is important to produce our results.
If not, we are going to depreciate remap in the Pytorch version of DHP.

Now you are ready to test offline-DHP.
Set ```mode = 'off_line'```, ```procedure = 'train'``` and ```if_log_results = False``` in ```config.py```, run following:
```bash
source ~/.bashrc
source activate dhp_env
python train.py
```

##### Test

Before testing offline-DHP, generate groundtruth heatmaps. Set ```mode = 'data_processor'``` and ```data_processor_id = 'generate_groundtruth_heatmaps'``` in ```config.py``` and run:
```bash
source ~/.bashrc
source activate dhp_env
python train.py
```

Now you are ready to test offline-DHP.
Note that the model is stored and restored automatically.
Thus, as long as you did not change the ```log_dir``` in ```config.py```, previous trained model will be restored.
Set ```mode = 'off_line'```, ```procedure = 'test'``` and ```if_log_results = True``` in ```config.py```, then run following:
```bash
source ~/.bashrc
source activate dhp_env
python train.py
```
The code will generate and store predicted_heatmaps, predicted_scanpath and CC value.

For results under more evaluation protocol. You may want to generate and store groundtruth_scanpaths with ```mode = 'data_processor'``` and ```data_processor_id = 'generate_groundtruth_scanpaths'```.

## Some hints on using the code.

* ```mode = 'data_processor'``` is a efficient way to process data under our TMUX manager, the code is in ```env_li.py```.

## Meet some issues?
Please don not hesitate to open an issue.

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
