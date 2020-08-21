# CUDA 10.1  Trial
Although the title of [the guide](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) was named Anaconda, however the guide was depending on miniconda. As a new developer to this domain I didn't catch the difference so I followed it without understanding.

## 1. Approach

After the [first trial](./trial-1-miniconda.md) and the failed outcome and after the review I was suspecting in the CUDA 11 version and that it needs to be downgraded specially because the documentation of TensorFlow indicates supported version is CUDA 10.1 

On the other hand and after the validation of TensorFlow, it is supporting at least Compute Compatibility 3.5 while my GPU is with Compute Compatibility 3.0 (a bit old laptop) so it is clear that I need to build TensorFlow from the source using docker image as per reference [^4]

After that I should create the environment using anaconda.  So the approach should be

## 2. Steps

When I started to follow the guidelines I found that the approach to setup the environment for using GPU that I have to do the following as per the documentation and as per reference [^1]

1. Uninstall CUDA
2. Install CUDA 10.1
3. Install cuDNN SDK 7.6.5 for CUDA 10.1
4. Build TensorFlow with Compute Compatibility 3.0
5. Install Miniconda latest and create the environment

### 2.1. Uninstall CUDA 

Here is the steps I followed to Uninstall CUDA as per [the documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-tk-and-driver) and the  other instructions in reference [^5]

```bash
#!/bin/bash

# To Remove CUDA Toolkit
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
sudo apt-get --purge remove libcublas10

sudo apt-get autoremove
sudo apt-get autoclean

# Remove any existing CUDA folders
sudo rm /etc/apt/sources.list.d/cuda*
sudo rm -rf /usr/local/cuda*
sudo rm /etc/apt/preferences.d/cuda-repository-pin-600 

```

### 2.2. Install CUDA v 10.1

I followed the steps and get input for System requirements for Ubuntu 20.04 & 18.04 for x86_64 architecture from [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-linux/index.html) as the following:

|Item | 20.04 | 18.04 | My environment Ubuntu 20.04.1 LTS |
|-----|-------|-------|------------|
| Kernel | 5.4.0 | 4.18.0 | 5.4.0-42-generic |
| Default GCC | 9.3.0 | 8.2.0 | gcc (Ubuntu 9.3.0-10ubuntu2) 9.3.0| 
| GLIBC | 2.31.0 | 2.28 |
| GCC | 9.x | - |
| ICC | 19.1 | 19.0 |
| PGI | 19.x, 20.x | 18.x, 19.x |
| XLC | NO | NO |
| CLANG | 9.0.0 | 8.0.0 |

The solution was to uninstall all cuda as per the above step and only install `nvidia-cuda-toolkit` as per the suggestion from the article [Installing TensorFlow GPU in Ubuntu 20.04](https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d). The two different methods provided by the article in resource [^1] and the CUDA documentation they are both providing directions to install CUDA 11.0 on Ubuntu 20.04 (the article) or CUDA 10.1 on Ubuntu 18.04 (CUDA Documentation). But I followed the last article is that it shows a way to install CUDA 10.1 but on Ubuntu 20.04 which is matching my case.

So I executed the following command.

```bash
sudo apt install nvidia-cuda-toolkit
```

when I run command

`nvidia-smi`

I got the following

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.05    Driver Version: 450.51.05    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro K1000M       On   | 00000000:01:00.0 Off |                  N/A |
| N/A   59C    P0    N/A /  N/A |    496MiB /  1999MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                              
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      5470      G   /usr/lib/xorg/Xorg                 39MiB |
|    0   N/A  N/A      7504      G   /usr/lib/xorg/Xorg                146MiB |
|    0   N/A  N/A      7772      G   /usr/bin/gnome-shell              147MiB |
|    0   N/A  N/A     34770      G   /proc/self/exe                    158MiB |
+-----------------------------------------------------------------------------+
```

and when I execute the command `nvcc -V` it shows me the following result:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

> **Note:** There is no relation between the CUDA version (11.0) in the outcome of `nvidia-smi` and the cuda version of `nvcc -V` outcome as [explained in this link](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi).

Then I added the following lines to the `~/.bashrc` to update the `PATH` variable as per the documentation as the following:

```bash
CUDA_PATH=/usr/lib/cuda
if [ -d "$CUDA_PATH" ] 
then
    export PATH="$CUDA_PATH/bin${PATH:+:${PATH}}"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="$CUDA_PATH/include${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset CUDA_PATH
```

then start the bash by executing command `source ~/.bashrc` and it is recommended to restart the machine for CUDA installation to complete.


#### 2.3.1 Failed trials but good experience !!

I executed the following commands for 20.04 

```bash
#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download .deb package
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub

# Installs all CUDA Toolkit packages required to develop CUDA applications.
# Does not include the driver.
sudo apt-get update
sudo apt-get -f install cuda-toolkit-10-1 cuda-libraries-10-1
sudo apt install nvidia-cuda-toolkit
```

> **NOTE :** Using command `sudo apt-get -y install cuda` as per the documentation will install the CUDA Toolkit and Driver packages and in our case it will be driver 418. It caused problem to my machine and I had to remove it following the guidelines in reference [^7]
> To overcome installing the drivers I used the meta package guidelines from reference [^6] to identify how to install the toolkit without installing the drivers. Sso now the command will be 
> `sudo apt-get -y install cuda-toolkit-10-1`

Then I added the following lines to the `~/.bashrc` to update the `PATH` variable as per the documentation as the following:

```bash
CUDA_PATH=/usr/local/cuda-10.1
if [ -d "$CUDA_PATH" ] 
then
    export PATH="$CUDA_PATH/bin${PATH:+:${PATH}}"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset CUDA_PATH
```

> **Note:** The documentation guidelines is to include path `/usr/local/cuda-10.1/NsightCompute-2019.1` to the `PATH` variable but as the folder was not created, I validated and as per [this community thread](https://forums.developer.nvidia.com/t/install-cuda10-0-1-on-ubuntu16-04-no-nsightcompute-2019-3-folder/81898/6), it is not needed

then start the bash by executing command `source ~/.bashrc` and it is recommended to restart the machine for CUDA installation to complete.

When I tried to execute the following command 

```bash
sudo apt install nvidia-cuda-toolkit
```

it failed due to missing dependency files on `libcublas10` after reviews, it was clear that I am trying to install overlapped library from two difference repositories as `nvidia-cuda-toolkit` is provided by Ubuntu repository while cuda-10.1 was provided by the deb packages.


### 2.3. Install cuDNN SDK 7.6

A list of downloadable cuDNN will be displayed > Expand the section for Download cuDNN **v7.6.5 (November 5th, 2019), for CUDA 10.1** (Here I made a mistake and used CUDA 10.2 instead of 10.1) then choose “cuDNN Library for Linux” to download cuDNN 7.6.5 for CUDA 10.1. After downloading cuDNN, extract the files by running,

```bash
tar -xvzf cudnn-10.1-linux-x64-v7.6.5.32.tgz
```
Next, copy the extracted files to the CUDA installation folder,

```bash
sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
```

Set the file permissions of cuDNN,

```bash
sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
```
### 2.4 Reboot

## 3. Outcome


## 4. Resources

1. [^1]Installing Tensorflow with CUDA & cuDNN GPU support on Ubuntu 20.04 and charge through your Linear Algebra calculations from [here](https://medium.com/@paulrohan/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a)

2. Install TensorFlow with GPU Support documentation from [here](https://www.tensorflow.org/install/gpu)

3. How to install the NVIDIA drivers on Ubuntu 20.04 Focal Fossa Linux from [here](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)

4. Build TensorFlow from source with GPU support from [here](https://www.tensorflow.org/install/source)

5. Installing CUDA 10.1 on Ubuntu 20.04 from [here](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)

6. CUDA Toolkit documentation - Meta Packages from [here](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-linux/index.html#package-manager-metas)

7. Black screen at boot after Nvidia driver installation on Ubuntu 18.04.2 LTS from [here](https://askubuntu.com/questions/1129516/black-screen-at-boot-after-nvidia-driver-installation-on-ubuntu-18-04-2-lts)