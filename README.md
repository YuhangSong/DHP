# DHP

This repo provides code for all the results reported in DHP paper. See [Modeling Attention in Panoramic Video: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1710.10755)

## Not ready yet!!

### cuda and cudnn
click cuda.deb
```
sudo apt-get -y install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev vim & echo "export PATH=/usr/local/cuda-8.0/bin:$PATH" >> /etc/profile & echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH" >> /etc/profile & sudo ldconfig & source /etc/profile & nvidia-smi & cd ../NVIDIA_CUDA-8.0_Samples & make -j40
cd ../rl_env
tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
sudo reboot
```


### for basic ff, entry rl_env
```

# not su root

sudo apt-get install openssh-server

sudo dpkg --add-architecture i386 && sudo apt-get update && sudo apt-get install libdbus-1-3:i386 libasound2:i386 libexpat1:i386 libfontconfig1:i386 libfreetype6:i386 libjpeg62:i386 libpng12-0:i386 libsm6:i386 libxdamage1:i386 libxext6:i386 libxfixes3:i386 libxinerama1:i386 libxrandr2:i386 libxrender1:i386 libxtst6:i386 zlib1g:i386 libc6:i386 && sudo dpkg -i teamviewer*.deb

# su root

sudo apt-get install aptitude -y && sudo apt-get -y install g++ && sudo apt-get -y install vim && sudo aptitude install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev git yasm libjpeg-turbo8-dev htop tmux && git config --global push.default "current"

bash Anaconda3-4.3.1-Linux-x86_64.sh

~/anaconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && ~/anaconda3/bin/conda config --set show_channel_urls yes && ~/anaconda3/bin/conda create -n song_1 python=2 -y && source activate song_1

source deactivate && source activate song_1 && pip install tensorflow-1.1.0-cp27-none-linux_x86_64.whl && tar -C /usr/local -xzf go1.7.4.linux-amd64.tar.gz && echo "export PATH=$PATH:/usr/local/go/bin" >> /etc/profile && source /etc/profile && source activate song_1 && cd gym && pip install -e .[atari] && cd .. && cd universe && pip install -e . && pip install matplotlib && cd .. && cd ffmpeg-3.2.4 && ./configure --enable-shared && make clean && make -j40 && make install && cd .. && cd opencv-2.4.13 && rm -r release && mkdir release && cd release/ && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && make clean && make -j40 && make install && cd ../../ && pip install torch-0.1.12.post2-cp27-none-linux_x86_64.whl && pip install torchvision && pip install lmdb scipy && cp remap ../ && chmod a+x ../remap &&

```

### for opencv, ffmpeg not supported
```
source activate song_1 && conda install -c menpo opencv=2 && source deactivate]
```

1，
   copy env/ff to your working dir

2，enter song_1

   pip install tensorflow... (1.1 version)
   
   cd gym
   
   pip install -e .[all] # all install

   tar -C /usr/local -xzf go1.7.4.linux-amd64.tar.
   
   gedit /etc/profile
   
   write following:
   
      export PATH=$PATH:/usr/local/go/bin
      
   source /etc/profile
   
   go version
   
   see if the version is 1.7, if not：
   
      which go # should show XXXX
      
      mv -r XXXX /home/
      
      source /etc/profile
      
      go version
      
      check again it the version is 1.7

   cd universe
   
   pip install -e .

   pip install matplotlib

3，

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
   put remap in home, chmod to X 
