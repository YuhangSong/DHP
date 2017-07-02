
# 简明版环境开始

简明版环境教程，严格遵守，非常容易。除非特殊说明可以同时进行的部分，一律按照顺序来

## 安装ubuntu

home装在最大的硬盘上一整个就好

用户名s，密码x

(以下需要的文件都在rl_env文件夹里)

先把env/rl_env复制到home下备用

## 安装显卡驱动

ctrl+alt+F1进入第一控制台
登录进入
sudo sh NVIDIA.run
各种默认继续，有一个地方，write x configuration file什么的，这里默认是no，选成yes即可
sudu reboot重启

## 安装teamviewer，一般双击即可

## 在普通用户下：sudo apt-get install openssh-server
sudo passwd
输入x
再次输入x
su root
输入x

## 此时，你已经可以远程了，用teamviewer或者ssh

## 以下操作在root下进行

### conda installation
```
bash Anaconda3-4.3.1-Linux-x86_64.sh && ~/anaconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && ~/anaconda3/bin/conda config --set show_channel_urls yes
```

### for basic ff, entry rl_env
```
~/anaconda3/bin/conda create -n song_1 python=2 -y && source activate song_1 && pip install tensorflow-1.1.0-cp27-none-linux_x86_64.whl && sudo aptitude install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig git Python-scipy htop tmux six txaio websocket docker g++ vim && tar -C /usr/local -xzf go1.7.4.linux-amd64.tar.gz && echo "export PATH=$PATH:/usr/local/go/bin" >> /etc/profile && source /etc/profile && source activate song_1 && cd gym && pip install -e .[atari] && cd .. && cd universe && pip install -e . && pip install matplotlib && cd .. && cd ffmpeg-3.2.4 && ./configure --enable-shared && make clean && make -j40 && make install && cd .. && pip install scipy && source deactivate
```

### for opencv, ffmpeg supported
```
cd opencv-2.4.13 && mkdir release && cd release/ && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && make clean && make -j40 && make install && cp lib/cv2.so ~/anaconda3/envs/song_1/include/
```

### for opencv, ffmpeg not supported
```
source activate song_1 && conda install -c menpo opencv=2 && source deactivate]
```

### for wgan
~/anaconda3/bin/conda create --name wgan_1 python=3.5 -y && source activate wgan_1 && pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl && pip install torchvision && pip install lmdb

python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda
python main.py --mlp_G --ngf 512

### for dowload lsun dataset
~/anaconda3/bin/conda create -n lsun_1 python=2.7 -y && source activate lsun_1 && git clone https://github.com/fyu/lsun.git && cd lsun && python2.7 download.py

sudo apt-get install aptitude

sudo aptitude install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig git Python-scipy htop tmux six txaio websocket docker g++ vim

这时，已经进入song_1虚拟环境（控制台会显示），如果没有，自行学习conda虚拟环境怎么用

下面1,2，3,4可以同时进行，两个不冲突

1，
   先把env/ff复制到工作目录下备用

2，进入虚拟环境！！！ 

   pip install tensorflow... (1.1 version)
   
   cd gym
   
   pip install -e .[all] # all install

   tar -C /usr/local -xzf go1.7.4.linux-amd64.tar.
   
   gedit /etc/profile
   
   写入以下内容：
   
      export PATH=$PATH:/usr/local/go/bin
      
   source /etc/profile
   
   go version
   
   看看上一步输出的版本是不是1.7，如果不是：
   
      which go # should show XXXX
      
      mv -r XXXX /home/
      
      source /etc/profile
      
      go version
      
      再次看看输出的版本是不是1.7

   cd universe
   
   pip install -e .

   pip install matplotlib

3，不要进入虚拟环境，在root下！！！ 

   tar ffmpeg
   
   cd ffmpeg
   
   ./configure --enable-shared
   
   make -j40
   
   make install
   
   unzip opencv-2.4.13.zip
   
   cd opencv-2.4.13
   
   mkdir release
   
   cd release/
   
   cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
   
   make -j40
   
   make install
   
   cp lib/cv2.so ~/anaconda3/lib
   
4,
   把remap放在home下，提升权限到可执行
   

==================================简明版完结=========================================





# GTN
Description: Generalization Tower Network (GTN)

Warning: this work is currently in submission for NIPS 2017.

###############################################################
#########################CT Agent##############################
###############################################################

***Enviriment***

    **install ubuntu**

        when boot with u disk, select uefi, for this result in able to install teamviewer
        part the disk yourself:
        1,logic, swap, 32g
        2,logic, ext4, /, for all rest
            (if error particpart) 3,logic, ext4, /boot,
            3, in lichen's computer, boot device select the second one,
               the one not installed the windows

   **teamviewer**
       syhdog@foxmail.com
       2281337833Song

    **nvidia(not recommended)**

        # sudo gedit /etc/modprobe.d/blacklist.conf

            add following:

            blacklist vga16fb

            blacklist nouveau

            blacklist rivafb

            blacklist nvidiafb

            blacklist rivatv

        # sudo apt-get remove --purge nvidia-*
          sudo apt-get remove --purge xserver-xorg-video-nouveau
        # sudo reboot
        # Ctrl + Alt +F1
            log in
        # sudo /etc/init.d/lightdm stop
        # sudo sh NVIDIA.run
        # sudo /etc/init.d/lightdm restart
        # sudo reboot

    **cuda(not recommended)**

        sudo sh cuda_8.0.44_linux.run # no for diver, yes for all others
        sudo apt-get -y install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
        sudo apt-get -y install vim
        export PATH=/usr/local/cuda-8.0/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

        sudo gedit /etc/profile #wirte flowing two
        export PATH=/usr/local/cuda-8.0/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

        sudo ldconfig #Enviriment varible take effect
        source /etc/profile

        nvidia-smi

        cd /root/NVIDIA_CUDA-8.0_Samples
        make -j8

        cd /home/s/RL/
        tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz
        sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
        sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
        sudo chmod a+r /usr/local/cuda/include/cudnn.h
        sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
        reboot

    **ness**
    sudo apt-get update
    sudo apt-get -y install build-essential

    sudo apt-get update && sudo apt-get upgrade
    sudo apt-get install -y linux-source
    sudo apt-get install -y linux-headers-`uname -r`

    **python**
    sudo apt-get install -y python-pip python-dev
    
    ============================================================================================
    
    bash Anaconda3-4.3.1-Linux-x86_64.sh 
    # add source from qinghua, so that the network is fixed
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --set show_channel_urls yes
    conda create -n song_1 python=2
    source activate song_1

    **tensorflow**
    pip install tensorflow... (1.1 version)
    	IF ERROR: .Exception: Versioning for this project requires either an sdist tarball, or access to an upstream git repository. Are you sure 	that git is installed?>>>
        $ sudo pip install --upgrade distribute

      to download tensorflow for any version:
        visit
          https://storage.googleapis.com/tensorflow
        find the version, e.t.
          linux/cpu/cloudml/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
        input in the address as:
          https://storage.googleapis.com/tensorflow/linux/cpu/cloudml/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
        to start download

    **gym**
    apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    apt-get -y install git
    cd gym
    pip install -e .[all] # all install
    cd ..
    
    ------>conda install -c https://conda.binstar.org/menpo opencv # not recommand

    **opencv**

     tar ffmpeg
     cd ffmpeg
     ./configure --enable-shared
     make -j40
     make install

     sudo apt-get -y install g++
     sudo apt-get -y install vim
     sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev
     unzip opencv-2.4.13.zip
     cd opencv-2.4.13
     mkdir release
     cd release/
     cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
     make -j16
     make install

    **universe**
        pip install six
        sudo apt-get install Python-scipy

        apt-get -y install htop
        apt-get -y install tmux

        sudo add-apt-repository ppa:ubuntu-lxc/lxd-stable
        sudo apt-get update
        sudo apt-get -y install golang libjpeg-turbo8-dev make

        sudo add-apt-repository ppa:ubuntu-lxc/lxd-stable && sudo apt-get update && sudo apt-get install golang

        tar -C /usr/local -xzf go1.7.4.linux-amd64.tar.gz
        gedit /etc/profile #write following
        export PATH=$PATH:/usr/local/go/bin
        source /etc/profile
        go version # see if go version is i.7
        #if go version is not 1.7
        which go # should show XXXX
        mv -r XXXX /home/
        source /etc/profile
        go version # to check if go version is 1.7

        pip install txaio
        pip install websocket
        pip install docker

        cd universe
        pip install -e .

        pip install matplotlib
        
        
    **set remap**
      put remap to home
      set it to excuatble with right click
      
    **set project**
      set ip and home in config
      
      配置ssh
      http://jingyan.baidu.com/article/9c69d48fb9fd7b13c8024e6b.html
      
      配置vim
      http://www.cnblogs.com/ma6174/archive/2011/12/10/2283393.html
      https://github.com/ma6174/vim
      
      配置文件列表显示
      http://blog.163.com/lgh_2002/blog/static/44017526201155113711656/
      ====================================================================
    **atom**
      su root

      sudo add-apt-repository ppa:webupd8team/atom
      sudo apt-get update
      sudo apt-get install atom


      #to uninstall atom

      sudo apt-get remove atom
      sudo add-apt-repository --remove ppa:webupd8team/atom


***Enviriment issues***

  IF ERROR: Pip install: ImportError: cannot import name IncompleteRead
    easy_install -U pip

  IF ERROR: after install ubuntu, can not find windows:
    sudo update-grub

  IF ERROR:
    .Exception: Versioning for this project requires either an sdist tarball, or access to an upstream git repository. Are you sure that git is installed?
      sudo pip install --upgrade distribute

  lcX (X<4) requires tensorflow 0.11

***issues when using remotes***

    **on agent server device**
      # if error accuors as : Resource temporarily unavailable
          # use $ 'ulimit -a' to check if there is any limit for this system,
            useful when you run this on a server
          # 'gedit /etc/security/limits.conf'
              add :
                          * soft noproc 65000
                          * hard noproc 65000
                          * soft nofile 1048576
                          * hard nofile 1048576
          # must reboot to take effect
          # before runing:
              ulimit -c 655350000000
              ulimit -c
              # it turns out the ct agent works fine without it, but the or agent still
                not working fine, i do not know why

'''

###############################################################
#########################Experiment############################
###############################################################'''

***update, issues and guessing***

* this version support lift and right consi layer
* disable right consi layer, for I found right consi maybe cause disadvantage
* I am not sure if consi_depth has a proper value, for now it seens to be not very sensitive
* It seens that multi lstm layer would not improve the agent, remain to be tested
* lrc seens to be damageful, and i have reason to believe it is true, so wandering
  if linear will do the job.

***good result record***

        * CT18goodserver >> l-t4-nd has produce usable result
        * CT37-2-good-server >> nl-nt4-dd has produce good result

***Experiment debug record:***

  CT51-ct-nl-nt4-dd-2
      tested with 8 games, 24 worker per device, tested to 18M do not have result
          guess: the lag is big, so test this problame

  CT51-ct-nl-nt4-dd-12
      tested with 2 games, 8 worker per device, to see if lag cause the bad performence

  CT51-ct-nl-nt4-dd-13
      the internet lag seens to be a big problame, test local run, should be as
      good as 37-2-good-server, or the project would be with bug

  CT51-ct-nl-nt4-dd-14
       the project seens to be bugged, so this version replace all code
       in a3c.py and moedl.py in this project with those in the 37-2-good-server

***structure name define***

  ct >> consi tower
  or >> orignal

  l  >> last layer allways lstm
  nl >> last layer not lstm

  t  >> tower structure
  nt >> squre structure
  num>> consi_depth

  nd >> dropout no discount
  dd >> dropout discount down (from low to high)
  du >> dropout discount up (from low to high)
  dc >> dropout with a constant 0.5

  gXwXsX >> X games and X workers per game, start with Xth game

    g2w10 >> (default)

    **Experiment of g2w10 only record whether some settings are demageful
      but not determine which is best yet, just prove which is ok
      to be tested further**

  nrc >> (default) no right consi layer
  lrc >>           lstm right consi layer
  frc >>           full-connected right consi layer

  lcX  >> lift consi layer has X conv layers
  llX  >> lift consi layer has X lstm layers
  dX   >> consi_depth is X

***waiting tested***

  **some settings i find promising or
    unsure if is good, ranked with my
    favour.**

  * CT52- l-nt4-nd-g8w4s0-1 >> on server
  * CT52-lc2-ll1-d4-nd-1

  * CT52- l- t4-dd-frc-1

  * CT52- l-nt4-dd-frc-1

  * CT52- l- t4-nd-g2w16s0-1

***testing***

**games are divided into sevarel of kinds**

env_seq_id = [

      'alien', 'amidar', 'bank_heist', 'ms_pacman', 'tutankham', 'venture', 'wizard_of_wor', # maze >> g7s0

      'assault', 'asteroids', 'beam_rider', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'atlantis', 'gravitar', 'phoenix', 'pooyan', 'riverraid', 'seaquest', 'space_invaders', 'star_gunner', 'time_pilot', 'zaxxon', 'yars_revenge', # shot 3 >> g18s7

      'asterix', 'elevator_action', 'berzerk', 'freeway', 'frostbite', 'journey_escape', 'kangaroo', 'krull', 'pitfall', 'skiing', 'up_n_down', 'qbert', 'road_runner', # advanture >> g13s25

      'double_dunk', 'ice_hockey', 'montezuma_revenge', 'gopher', # iq >> g4s38

      'breakout', 'pong', 'private_eye', 'tennis', 'video_pinball', # pong >> g5s42

      'fishing_derby', 'name_this_game', # fishing >> g2s47

      'bowling', # bowing >> g1s49

      'battle_zone', 'boxing', 'jamesbond', 'robotank', 'solaris', # shot 1 >> g5s50

      'enduro', # drive 1 >> g1s55

      'kung_fu_master' # fight >> g1s56

  ]

  * CT52-lc3-ll1-d4-nd-g7s0w2-1 >> s-2
  * CT52-lc3-ll1-d4-nd-g18s7w2-1 >> server
  * CT52-lc3-ll1-d4-nd-g13s25w2-1 >> yuhangsong
  * CT52-lc3-ll1-d4-nd-g4s38w2-1 >> s
  * CT52-lc3-ll1-d4-nd-g5s42w2-1 >> tianyi
  * CT52-lc3-ll1-d4-nd-g5s50w2-1 >> s-1


***good result***

  **if the result are set in the same block,
    it means i can't tell which is really
    better, they are similiar.**

  * result: R4-lc3-ll1-d4-nd-g18s7w2-3  project: R4-lc3-ll1-d4-nd-g18s7w2-1  the consi feature is treat as observation, trained upon what is gained at the interacting time
  * testing\\ result: R5-lc3-ll1-d4-nd-g18s7w2-2  project: R5-lc3-ll1-d4-nd-g18s7w2-2  the consi feature is treat as observation, trained upon what is gained at the training time

  * result=ff11-t2-gamma-099-final_discount-4 project=ff11-t2-gamma-099-final_discount-4
    **bug on the speed, not imlement the scaler**


  * result=ff12-offline-feild-gamma-000-finaldiscount-4 project=ff12-offline-feild-gamma-000-finaldiscount-4
  * result=ff12-offline-feild-gamma-099-finaldiscount-4 project=ff12-offline-feild-gamma-099-finaldiscount-4 (good)

  * CT52-lc3-ll1-d4-nd-g8w5s0-1   (promising)

  * CT52-lc3-ll1-d4-nd-1    >> on song pc (promising)
  * CT52- l- t4-nd-1

  * CT52-lc3-ll1-d4-dd-1 (promising)
  * CT52- l-nt4-nd-1 (promising)
  * CT52- l- t4-nd-frc-4 (promising, but bad)

  * CT52- l- t4-dd-1 (promising)
  * CT52-nl-nt4-dd-1 (CT37-2)
  * CT52-nl- t4-nd-lrc-1 (promising)

***bad result(crushed or never solved, unranked)***

  * CT52-lc3-ll1-d4-nd-frc-1

  * CT52-lc1-ll1-d4-nd-1 # too much memory cost for the link of conv layer
  * CT52-lc2-ll1-d4-nd-1 # too much memory cost for the link of conv layer

  * CT52-nl- t4-dd-lrc-1
  * CT52- l- t4-dd-frc-g8w4s0-1
  * CT52- l- t4-dd-frc-1

  * CT52- l- t4-du-1
  * CT52- l- t4-dc-1

  * CT52-nl-nt4-nd-lrc-4

###############################################################
#########################Remote Enviriment#####################
###############################################################

***Enviriment***

  sudo apt-get install apt-transport-https ca-certificates
  sudo apt-key adv \
                 --keyserver hkp://ha.pool.sks-keyservers.net:80 \
                 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
  # If the above keyserver is not available, try hkp://pgp.mit.edu:80 or hkp://keyserver.ubuntu.com:80.

  echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
  sudo apt-get update
  apt-cache policy docker-engine

  sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual

  sudo apt-get install linux-image-generic-lts-trusty

  sudo apt-get install docker-engine # unstable, if not moving, Ctrl+C, try more times
  sudo service docker start
  sudo docker run hello-world

  docker run -p 5917:5900 -p 15917:15900 quay.io/openai/universe.gym-core:0.20.0 # unstable, if not moving, Ctrl+C, try more times

***before run***

  if just open the compute, sudo service docker start
  set if-testing-env to true to see if all device remotes works fine


###############################################################
#########################Some other notice#####################
###############################################################
gym version may cause difference in:
  env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
  env.spec.timestep_limit
  find in a3c.py

  192.168.226.119


***update to tensorflow 1.0***
pip uninstall tensorflow
pip install --upgrade tensorflow-gpu

***for ff***
#install ffmpeg
tar
./configure
make
make install

#if error：videoCapture failed
#openvc require ffmege to open mp4
tar
./configure --enable-shared
make
make install
+reinstall opencv

#remap
put remap to home


#ff result
x1 for cc_count_to is best
use ff-test-randomwalk on worker every batch 8 to 12, reset exp name for every run， plaste the sequence at test comun
use ff-test-model on worker every batch 8-12, set to model, plaste the sequence at test comun
use ff-test-x1-center-bias on my pc every batch 24, plaste the sequence at test comun

f15 3rd copy is the recent run for 3 days on server

ff40 and result/ff40(copy) on server is currently working
ff40 on worker has
    tf.Session(server.target, config=config).run(tf.global_variables_initializer())
    so that all workers would worker, if not, some worker would die, not happened on server, which is the cluster_main
    with it, the model's global step is not load, but donot know if any further effect>>the model is reseted..
ff40 on serve donnot have obove and works fine to reload model

continuous use a tanh to produce mu, not lineer in the paper
the cost of policy is in question for it use sigma_sq as trainable
my_sigma is not right

to learn and setup github on ubuntu
http://blog.csdn.net/tina_ttl/article/details/51326684
