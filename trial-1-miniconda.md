# Anaconda Trial
Although the title of [the guide](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) was named Anaconda, however the guide was depending on miniconda. As a new developer to this domain I didn't catch the difference so I followed it without understanding.

## 1. Approach

Using Anaconda consists of the following:

1. Install miniconda on your computer
2. Create a new `conda` environment using [this project](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)
3. Each time you wish to work, activate your `conda` environment.

But I am afraid that this wish never comes to true that way :). Any way before that, you will need to install the needed software for GPU support so you can use TensorFlow with GPU.

## 2. Steps

When I started to follow the guidelines I found that the approach to setup the environment for using GPU that I have to do the following as per the documentation and as per reference [^1]

1. Install NVIDIA GPU Drivers
2. Install CUDA (v 11.0 as per the reference)
3. Install cuDNN SDK 7.6
4. Install Miniconda latest and create the environment

### 2.1. Install NVIDIA GPU Drivers

I upgraded my environment drivers following reference [^3] to `version 450.51.05`

### 2.2. Install CUDA (v 11.0 as per the reference)

I followed reference [^1] to install CUDA v 11.0 which was just release and it supports Ubuntu 20.04. Although the guidelines for TensorFlow in Reference [^2] was indicating that the supported level was CUDA v10.1 but it seems that the article helped with it.

For System requirements for Ubuntu 20.04 & 18.04 for x86_64 architecture from [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-linux/index.html) as the following:

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

Then I executed the following commands for 20.04 

```bash
#!/bin/bash

# Prepare for installation
sudo apt-get install linux-headers-$(uname -r)

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install nvidia-cuda-toolkit
```

when I run command 

> `nvidia-smi`

I got the following 

> ```
> +-----------------------------------------------------------------------------+
> | NVIDIA-SMI 450.51.05    Driver Version: 450.51.05    CUDA Version: 11.0     |
> |-------------------------------+----------------------+----------------------+
> | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
> | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
> |                               |                      |               MIG M. |
> |===============================+======================+======================|
> |   0  Quadro K1000M       On   | 00000000:01:00.0 Off |                  N/A |
> | N/A   59C    P0    N/A /  N/A |    496MiB /  1999MiB |      0%      Default |
> |                               |                      |                  N/A |
> +-------------------------------+----------------------+----------------------+
>                                                                               
> +-----------------------------------------------------------------------------+
> | Processes:                                                                  |
> |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
> |        ID   ID                                                   Usage      |
> |=============================================================================|
> |    0   N/A  N/A      5470      G   /usr/lib/xorg/Xorg                 39MiB |
> |    0   N/A  N/A      7504      G   /usr/lib/xorg/Xorg                146MiB |
> |    0   N/A  N/A      7772      G   /usr/bin/gnome-shell              147MiB |
> |    0   N/A  N/A     34770      G   /proc/self/exe                    158MiB |
> +-----------------------------------------------------------------------------+
> ```

### 2.3. Install cuDNN SDK 7.6

A list of downloadable cuDNN will be displayed > Expand the section for Download cuDNN **v7.6.5 (November 5th, 2019), for CUDA 10.1** (Here I made a mistake and used CUDA 10.2 instead of 10.1)

### 2.4. Install Miniconda / Anaconda and create the environment

*I repeated the below for both miniconda and anaconda and I got the same failed results*

1. I downloaded the [`Miniconda3-latest-Linux-x86_64.sh`](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) for Python 3.8
2. In my terminal I ran the command 

    > `bash Miniconda3-latest-Linux-x86_64.sh`

    which added a path to the miniconda default folder to the `PATH` bash variable of ~/.bash_profile

3. create the environment in the miniconda like the following

    > ```bash
    > #!/bin/bash
    > # 3.1. Get the repository
    > git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git
    >
    > # 3.2. Create the environment
    > cd CarND-Term1-Starter-Kit
    > conda env create -f environment-gpu.yml
    >
    > # 3.3. Activate the repository
    > activate carnd-term1
    > ```

## 3. Outcome

```bash
Warning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I am adding one for you, but still nagging you.
Collecting package metadata (repodata.json): done
Solving environment: - 
Found conflicts! Looking for incompatible packages.
This can take several minutes.  Press CTRL-C to abort.
failed                                                                                                                                                     \  
Solving environment: \ 
Found conflicts! Looking for incompatible packages.
This can take several minutes.  Press CTRL-C to abort.
failed                                                                                                                                                     \  

UnsatisfiableError: The following specifications were found to be incompatible with each other:

Output in format: Requested package -> Available versions

Package pandas conflicts for:
seaborn -> statsmodels[version='>=0.8.0'] -> pandas[version='>=0.14|>=0.21']
seaborn -> pandas[version='>=0.14.0|>=0.22.0']
pandas
scikit-image -> dask[version='>=0.5'] -> pandas[version='>=0.18.0|>=0.19.0|>=0.21.0|>=0.23.0']

Package matplotlib conflicts for:
scikit-image -> matplotlib[version='>=1.1|>=1.3.1|>=2.0.0']
seaborn -> matplotlib[version='>=1.4.3|>=2.1.2']
matplotlib

Package pillow conflicts for:
seaborn -> matplotlib-base[version='>=2.1.2'] -> pillow[version='>=6.2.0']
pillow
imageio -> pillow
scikit-image -> matplotlib-base[version='>=2.0.0'] -> pillow[version='>=6.2.0']
scikit-image -> pillow[version='>=1.7.8|>=2.1.0|>=4.3.0']
matplotlib -> matplotlib-base[version='>=3.3.1,<3.3.2.0a0'] -> pillow[version='>=6.2.0']

Package ld_impl_linux-64 conflicts for:
matplotlib -> python[version='>=3.8,<3.9.0a0'] -> ld_impl_linux-64[version='>=2.34']
numpy -> python[version='>=3.6,<3.7.0a0'] -> ld_impl_linux-64[version='>=2.34']
h5py -> python[version='>=3.6,<3.7.0a0'] -> ld_impl_linux-64[version='>=2.34']
scikit-image -> python[version='>=3.6,<3.7.0a0'] -> ld_impl_linux-64[version='>=2.34']
imageio -> python=2.7 -> ld_impl_linux-64[version='>=2.34']
pip -> python[version='>=3'] -> ld_impl_linux-64[version='>=2.34']
flask-socketio -> python -> ld_impl_linux-64[version='>=2.34']
pandas -> python[version='>=3.6,<3.7.0a0'] -> ld_impl_linux-64[version='>=2.34']
pillow -> python[version='>=3.8,<3.9.0a0'] -> ld_impl_linux-64[version='>=2.34']
eventlet -> python[version='>=3.8,<3.9.0a0'] -> ld_impl_linux-64[version='>=2.34']
pyqt=4.11.4 -> python=3.6 -> ld_impl_linux-64[version='>=2.34']
scipy -> python[version='>=3.6,<3.7.0a0'] -> ld_impl_linux-64[version='>=2.34']
seaborn -> python[version='>=3.6'] -> ld_impl_linux-64[version='>=2.34']
scikit-learn -> python[version='>=3.7,<3.8.0a0'] -> ld_impl_linux-64[version='>=2.34']
jupyter -> python -> ld_impl_linux-64[version='>=2.34']

Package gdbm conflicts for:
pillow -> pypy3.6[version='>=7.3.1'] -> gdbm[version='>=1.18,<1.19.0a0']
pandas -> pypy3.6[version='>=7.3.1'] -> gdbm[version='>=1.18,<1.19.0a0']
scipy -> pypy3.6[version='>=7.3.1'] -> gdbm[version='>=1.18,<1.19.0a0']
eventlet -> pypy3.6[version='>=7.3.1'] -> gdbm[version='>=1.18,<1.19.0a0']
numpy -> pypy3.6[version='>=7.3.1'] -> gdbm[version='>=1.18,<1.19.0a0']

Package ffmpeg conflicts for:
imageio -> ffmpeg=2.7
ffmpeg

Package icu conflicts for:
matplotlib -> icu[version='>=58.2,<59.0a0']
pyqt=4.11.4 -> qt[version='>=4.8.6,<5.0'] -> icu=58
scikit-image -> matplotlib-base[version='>=2.0.0'] -> icu[version='>=58.2,<59.0a0|>=64.2,<65.0a0|>=67.1,<68.0a0']
matplotlib -> matplotlib-base[version='>=3.3.0,<3.3.1.0a0'] -> icu[version='>=64.2,<65.0a0|>=67.1,<68.0a0']
seaborn -> matplotlib-base[version='>=2.1.2'] -> icu[version='>=58.2,<59.0a0|>=64.2,<65.0a0|>=67.1,<68.0a0']

Package setuptools conflicts for:
python==3.5.2 -> pip -> setuptools
matplotlib -> setuptools
pip -> setuptools
scikit-learn -> joblib[version='>=0.11'] -> setuptools
seaborn -> matplotlib-base[version='>=2.1.2'] -> setuptools
scikit-image -> matplotlib-base[version='>=2.0.0'] -> setuptools

Package openblas-devel conflicts for:
scikit-learn -> openblas[version='>=0.3.3,<0.3.4.0a0'] -> openblas-devel[version='0.3.3|>=0.2.20,<0.2.21.0a0',build='2|3|1']
scipy -> openblas[version='>=0.3.3,<0.3.4.0a0'] -> openblas-devel[version='0.3.3|>=0.2.20,<0.2.21.0a0',build='2|3|1']
numpy -> openblas[version='>=0.3.3,<0.3.4.0a0'] -> openblas-devel[version='0.3.10|0.3.3|>=0.2.20,<0.2.21.0a0|0.3.6|0.3.6|0.3.6|>=0.3.2,<0.3.3.0a0',build='0|0|2|3|1|2|1']

Package pip conflicts for:
pyqt=4.11.4 -> python=3.6 -> pip
pip
eventlet -> python[version='>=3.8,<3.9.0a0'] -> pip
matplotlib -> python[version='>=3.8,<3.9.0a0'] -> pip
scikit-learn -> python[version='>=3.7,<3.8.0a0'] -> pip
seaborn -> python[version='>=3.6'] -> pip
jupyter -> python -> pip
scikit-image -> python[version='>=3.6,<3.7.0a0'] -> pip
imageio -> python=3.5 -> pip
numpy -> python[version='>=3.6,<3.7.0a0'] -> pip
h5py -> python[version='>=3.6,<3.7.0a0'] -> pip
pandas -> python[version='>=3.6,<3.7.0a0'] -> pip
flask-socketio -> python -> pip
scipy -> python[version='>=3.6,<3.7.0a0'] -> pip
python==3.5.2 -> pip

Package six conflicts for:
eventlet -> six[version='>=1.10.0']
h5py -> unittest2 -> six[version='>=1.4']
eventlet -> pyopenssl -> six[version='>=1.5.2']
pandas -> python-dateutil[version='>=2.7.3'] -> six[version='>=1.5']
seaborn -> patsy -> six
numpy -> mkl-service[version='>=2,<3.0a0'] -> six
scikit-learn -> mkl-service[version='>=2,<3.0a0'] -> six
flask-socketio -> python-socketio[version='>=4.3.0'] -> six[version='>=1.9.0']
matplotlib -> cycler -> six[version='>=1.5']
scikit-image -> six[version='>=1.4|>=1.7.3']
h5py -> six
scipy -> mkl-service[version='>=2,<3.0a0'] -> six

Package libpng conflicts for:
seaborn -> matplotlib-base[version='>=2.1.2'] -> libpng[version='>=1.6.23,<1.7|>=1.6.32,<1.7.0a0|>=1.6.34,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.37,<1.7.0a0|>=1.6.36,<1.7.0a0']
pillow -> freetype[version='>=2.9.1,<3.0a0'] -> libpng[version='1.6.*|>=1.6.21,<1.7|>=1.6.22,<1.6.31|>=1.6.32,<1.6.35|>=1.6.34,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.37,<1.7.0a0|>=1.6.32,<1.7.0a0|>=1.6.28,<1.7|>=1.6.23,<1.7']
scikit-image -> matplotlib-base[version='>=2.0.0'] -> libpng[version='>=1.6.23,<1.7|>=1.6.32,<1.7.0a0|>=1.6.34,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.37,<1.7.0a0|>=1.6.36,<1.7.0a0']
matplotlib -> libpng[version='>=1.6.23,<1.7|>=1.6.37,<1.7.0a0|>=1.6.36,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.34,<1.7.0a0|>=1.6.32,<1.7.0a0']
matplotlib -> freetype=2.6 -> libpng[version='1.6.*|>=1.6.21,<1.7|>=1.6.32,<1.6.35']
pyqt=4.11.4 -> qt[version='>=4.8.6,<5.0'] -> libpng[version='>=1.6.28,<1.7|>=1.6.36,<1.7.0a0']
ffmpeg -> freetype[version='>=2.9.1,<3.0a0'] -> libpng[version='>=1.6.32,<1.6.35|>=1.6.34,<1.7.0a0|>=1.6.35,<1.7.0a0|>=1.6.37,<1.7.0a0|>=1.6.32,<1.7.0a0']

Package scipy conflicts for:
seaborn -> statsmodels[version='>=0.8.0'] -> scipy[version='>=0.14|>=1.0']
seaborn -> scipy[version='>=0.15.2|>=1.0.1']
scikit-learn -> scipy
scikit-image -> scipy[version='>=0.17|>=0.19|>=0.9']
scipy

Package pyqt conflicts for:
matplotlib -> pyqt[version='4.11.*|>=5.12.3,<5.13.0a0|>=5.6.0,<5.7.0a0|>=5.9.2,<5.10.0a0|5.9.*|>=5.6,<6.0a0|5.*|5.6.*']
jupyter -> qtconsole -> pyqt[version='4.11.*|5.6.*|>=5.6.0,<5.7.0a0|>=5.9.2,<5.10.0a0']
scikit-image -> matplotlib[version='>=2.0.0'] -> pyqt[version='4.11.*|5.*|5.9.*|>=5.12.3,<5.13.0a0|>=5.6.0,<5.7.0a0|>=5.9.2,<5.10.0a0|>=5.6,<6.0a0|5.6.*']
pyqt=4.11.4
seaborn -> matplotlib[version='>=2.1.2'] -> pyqt[version='4.11.*|5.*|5.9.*|>=5.12.3,<5.13.0a0|>=5.6.0,<5.7.0a0|>=5.9.2,<5.10.0a0|>=5.6,<6.0a0|5.6.*']

Package numpy conflicts for:
scipy -> numpy[version='1.10.*|1.11.*|1.12.*|1.13.*|>=1.11|>=1.11.3,<2.0a0|>=1.14.6,<2.0a0|>=1.16.5,<2.0a0|>=1.18.5,<2.0a0|>=1.18.1,<2.0a0|>=1.9.3,<2.0a0|>=1.9|>=1.15.1,<2.0a0']
scikit-image -> numpy[version='1.10.*|1.11.*|1.12.*|1.13.*|>=1.11|>=1.11.3,<2.0a0|>=1.14.6,<2.0a0|>=1.9.3,<2.0a0|>=1.13.3,<2.0a0']
imageio -> numpy
seaborn -> numpy[version='>=1.13.3|>=1.9.3']
scikit-image -> imageio[version='>=2.3.0'] -> numpy[version='>=1.10|>=1.10.4|>=1.11.0|>=1.11.3|>=1.15.1|>=1.15.1,<2.0a0|>=1.15.4,<2.0a0|>=1.16.5,<2.0a0|>=1.18.5,<2.0a0|>=1.18.1,<2.0a0|>=1.8|>=1.9|>=1.13.0']
scikit-learn -> scipy -> numpy[version='>=1.11|>=1.18.1,<2.0a0|>=1.18.5,<2.0a0|>=1.15.1,<2.0a0']
pandas -> numpy[version='1.10.*|1.11.*|1.12.*|1.13.*|>=1.11|>=1.11.*|>=1.12.1,<2.0a0|>=1.14.6,<2.0a0|>=1.15.4,<2.0a0|>=1.16.5,<2.0a0|>=1.18.5,<2.0a0|>=1.18.4,<2.0a0|>=1.18.1,<2.0a0|>=1.9.3,<2.0a0|>=1.9.*|>=1.9|>=1.8|>=1.7|>=1.16.6,<2.0a0|>=1.13.3,<2.0a0|>=1.11.3,<2.0a0']
numpy
h5py -> numpy[version='1.10.*|1.11.*|1.12.*|1.13.*|>=1.14.6,<2.0a0|>=1.9.3,<2.0a0|>=1.8|>=1.8,<1.14|>=1.11.3,<2.0a0']
matplotlib -> matplotlib-base[version='>=3.3.1,<3.3.2.0a0'] -> numpy[version='>=1.11.3,<2.0a0|>=1.15.4,<2.0a0|>=1.16.5,<2.0a0|>=1.18.5,<2.0a0|>=1.9.3,<2.0a0']
seaborn -> statsmodels[version='>=0.8.0'] -> numpy[version='1.10.*|1.11.*|1.12.*|1.13.*|>=1.11.*|>=1.11.3,<2.0a0|>=1.11|>=1.14.6,<2.0a0|>=1.15.4,<2.0a0|>=1.16.5,<2.0a0|>=1.18.1,<2.0a0|>=1.17.5,<2.0a0|>=1.9.3,<2.0a0|>=1.18.5,<2.0a0|>=1.18.4,<2.0a0|>=1.12.1,<2.0a0|>=1.9.*|>=1.16.6,<2.0a0|>=1.13.3,<2.0a0|>=1.9|>=1.15.1,<2.0a0|>=1.8|>=1.7|>=1.4.0']
matplotlib -> numpy[version='1.10.*|1.11.*|>=1.14.6,<2.0a0']
scikit-learn -> numpy[version='1.10.*|1.11.*|1.12.*|1.13.*|>=1.11.3,<2.0a0|>=1.14.6,<2.0a0|>=1.16.5,<2.0a0|>=1.9.3,<2.0a0|>=1.9']

Package openssl conflicts for:
imageio -> python=3.5 -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1e,<1.1.2a']
seaborn -> python[version='>=3.6'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
scikit-learn -> python[version='>=3.7,<3.8.0a0'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
jupyter -> python -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
ffmpeg -> openssl[version='>=1.1.1d,<1.1.2a']
flask-socketio -> python -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
pyqt=4.11.4 -> python=3.6 -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.0.2r,<1.0.3a|>=1.1.1b,<1.1.2a']
pip -> python[version='>=3'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
python==3.5.2 -> openssl=1.0
pillow -> python[version='>=3.8,<3.9.0a0'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
numpy -> pypy3.6[version='>=7.3.1'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1b,<1.1.2a']
scikit-image -> python[version='>=3.6,<3.7.0a0'] -> openssl[version='!=1.1.1e|1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1b,<1.1.2a']
matplotlib -> python[version='>=3.8,<3.9.0a0'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a']
h5py -> python[version='>=3.6,<3.7.0a0'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1b,<1.1.2a']
scipy -> pypy3.6[version='>=7.3.1'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1b,<1.1.2a']
eventlet -> python[version='>=3.8,<3.9.0a0'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1b,<1.1.2a']
pandas -> python[version='>=3.6,<3.7.0a0'] -> openssl[version='1.0.*|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1f,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.0.2n,<1.0.3a|>=1.0.2m,<1.0.3a|>=1.0.2l,<1.0.3a|>=1.1.1b,<1.1.2a']

Package imageio conflicts for:
imageio
scikit-image -> imageio[version='>=2.1.0|>=2.3.0']

Package freetype conflicts for:
ffmpeg -> freetype[version='2.8.1|2.8.1.*|>=2.8.1,<2.8.2.0a0|>=2.8.1,<2.9.0a0|>=2.9.1,<3.0a0|>=2.8,<2.9.0a0']
matplotlib -> freetype[version='2.6.*|>=2.9.1,<3.0a0|>=2.8,<2.9.0a0']
seaborn -> matplotlib-base[version='>=2.1.2'] -> freetype[version='2.6.*|>=2.8,<2.9.0a0|>=2.9.1,<3.0a0']
imageio -> pillow -> freetype[version='2.5.*|2.6.*|2.7|2.7.*|2.7|2.8.*|2.8.1|2.8.1.*|>=2.8.1,<2.9.0a0|>=2.9.1,<3.0a0|>=2.8,<2.9.0a0']
pyqt=4.11.4 -> qt[version='>=4.8.6,<5.0'] -> freetype[version='2.7|2.7.*']
pillow -> freetype[version='2.5.*|2.6.*|2.7|2.7.*|2.7|2.8.*|2.8.1|2.8.1.*|>=2.8.1,<2.9.0a0|>=2.9.1,<3.0a0|>=2.8,<2.9.0a0']
scikit-image -> matplotlib-base[version='>=2.0.0'] -> freetype[version='2.5.*|2.6.*|2.7|2.8.*|2.8.1|2.8.1.*|>=2.8.1,<2.9.0a0|>=2.9.1,<3.0a0|>=2.8,<2.9.0a0|2.7|2.7.*']

Package libtiff conflicts for:
pillow -> libtiff[version='4.*|4.0.*|>=4.0.10,<5.0a0|>=4.1.0,<5.0a0|>=4.0.9,<5.0a0|>=4.0.8,<4.0.10|>=4.0.3,<4.0.8|4.0.6|>=4.0.8,<5.0a0']
imageio -> pillow -> libtiff[version='4.*|4.0.*|>=4.0.10,<5.0a0|>=4.1.0,<5.0a0|>=4.0.9,<5.0a0|>=4.0.8,<4.0.10|>=4.0.3,<4.0.8|4.0.6|>=4.0.8,<5.0a0']
scikit-image -> pillow[version='>=4.3.0'] -> libtiff[version='4.*|4.0.*|>=4.0.10,<5.0a0|>=4.1.0,<5.0a0|>=4.0.9,<5.0a0|>=4.0.8,<4.0.10|>=4.0.3,<4.0.8|4.0.6|>=4.0.8,<5.0a0']
pyqt=4.11.4 -> qt[version='>=4.8.6,<5.0'] -> libtiff[version='4.0.*|>=4.0.10,<5.0a0']

Package enum34 conflicts for:
imageio -> enum34
scikit-image -> imageio[version='>=2.3.0'] -> enum34
matplotlib -> pyqt -> enum34
eventlet -> enum34

Package zlib conflicts for:
ffmpeg -> gnutls=3.5 -> zlib==1.2.8
ffmpeg -> zlib[version='1.2.*|1.2.11|1.2.11.*|>=1.2.11,<1.3.0a0']

Package jinja2 conflicts for:
jupyter -> nbconvert -> jinja2
flask-socketio -> flask[version='>=0.9'] -> jinja2[version='>=2.10|>=2.10.1|>=2.4']

Package olefile conflicts for:
imageio -> pillow -> olefile
pillow -> olefile
scikit-image -> pillow[version='>=4.3.0'] -> olefile

Package _libgcc_mutex conflicts for:
numpy -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
scikit-learn -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
ffmpeg -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
h5py -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
pillow -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
scikit-image -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
pandas -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
matplotlib -> libgcc-ng[version='>=7.3.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']
scipy -> libgcc-ng[version='>=7.5.0'] -> _libgcc_mutex[version='*|0.1',build='conda_forge|main']

Package expat conflicts for:
pandas -> pypy3.6[version='>=7.3.1'] -> expat[version='>=2.2.9,<2.3.0a0']
numpy -> pypy3.6[version='>=7.3.1'] -> expat[version='>=2.2.9,<2.3.0a0']
pillow -> pypy3.6[version='>=7.3.1'] -> expat[version='>=2.2.9,<2.3.0a0']
pyqt=4.11.4 -> qt[version='>=4.8.6,<5.0'] -> expat[version='>=2.2.5,<2.3.0a0']
scipy -> pypy3.6[version='>=7.3.1'] -> expat[version='>=2.2.9,<2.3.0a0']
eventlet -> pypy3.6[version='>=7.3.1'] -> expat[version='>=2.2.9,<2.3.0a0']

Package futures conflicts for:
imageio -> futures
scikit-image -> imageio[version='>=2.3.0'] -> futures
matplotlib -> tornado -> futures

Package wheel conflicts for:
python==3.5.2 -> pip -> wheel
pip -> wheel

Package bzip2 conflicts for:
jupyter -> python -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
seaborn -> python[version='>=3.6'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
pyqt=4.11.4 -> python=3.6 -> bzip2[version='>=1.0.6,<2.0a0']
scipy -> pypy3.6[version='>=7.3.1'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
numpy -> pypy3.6[version='>=7.3.1'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
matplotlib -> python[version='>=3.7,<3.8.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
scikit-learn -> python[version='>=3.7,<3.8.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
ffmpeg -> bzip2[version='1.0.*|>=1.0.6,<1.1.0a0|>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
scikit-image -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
pip -> python[version='>=3'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
imageio -> python=3.5 -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
h5py -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
pandas -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
flask-socketio -> python -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
pillow -> python[version='>=3.7,<3.8.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
eventlet -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']

Package pypy3.6 conflicts for:
pillow -> python[version='>=3.6,<3.7.0a0'] -> pypy3.6[version='7.3.*|7.3.0.*|7.3.1.*']
pillow -> pypy3.6[version='>=7.3.1']

Package pytz conflicts for:
pandas -> pytz[version='>=2017.2']
scikit-image -> matplotlib-base[version='>=2.0.0'] -> pytz
matplotlib -> pytz
seaborn -> matplotlib-base[version='>=2.1.2'] -> pytz[version='>=2017.2']

Package tornado conflicts for:
jupyter -> ipykernel -> tornado[version='>=4|>=4,<6|>=4.0|>=4.2|>=5.0|>=5.0,<7|>=4.1,<7']
seaborn -> matplotlib-base[version='>=2.1.2'] -> tornado
scikit-image -> matplotlib-base[version='>=2.0.0'] -> tornado
matplotlib -> tornado

Package libgcc conflicts for:
seaborn -> scipy[version='>=0.15.2'] -> libgcc
matplotlib -> pyqt -> libgcc
scipy -> libgcc
scikit-learn -> scipy -> libgcc
scikit-image -> scipy[version='>=0.17'] -> libgcc

Package qt conflicts for:
matplotlib -> pyqt -> qt[version='4.8.*|5.6.*|5.9.*|>=5.12.5,<5.13.0a0|>=5.9.7,<5.10.0a0|>=5.6.2,<5.7.0a0|>=4.8.6,<5.0|>=5.9.6,<5.10.0a0|>=5.9.4,<5.10.0a0|>=5.6.3,<5.7.0a0']
pyqt=4.11.4 -> qt[version='4.8.*|>=4.8.6,<5.0']

Package blis conflicts for:
numpy -> libblas[version='>=3.8.0,<4.0a0'] -> blis[version='0.5.1.*|>=0.5.2,<0.5.3.0a0|>=0.6.0,<0.6.1.0a0|>=0.6.1,<0.6.2.0a0|>=0.7.0,<0.7.1.0a0']
scipy -> libblas[version='>=3.8.0,<4.0a0'] -> blis[version='0.5.1.*|>=0.5.2,<0.5.3.0a0|>=0.6.0,<0.6.1.0a0|>=0.6.1,<0.6.2.0a0|>=0.7.0,<0.7.1.0a0']

Package gmp conflicts for:
ffmpeg -> gnutls=3.5 -> gmp=6.1
ffmpeg -> gmp[version='>=6.1.2|>=6.1.2,<7.0a0|>=6.2.0,<7.0a0']
scipy -> libgcc -> gmp[version='>=4.2']

Package packaging conflicts for:
pip -> wheel -> packaging[version='>=20.2']
scikit-image -> pooch[version='>=0.5.2'] -> packaging

Package certifi conflicts for:
pip -> setuptools -> certifi[version='>=2016.09|>=2016.9.26']
matplotlib -> setuptools -> certifi[version='>=2016.09|>=2016.9.26']The following specifications were found to be incompatible with your CUDA driver:

  - feature:/linux-64::__cuda==11.0=0
  - feature:|@/linux-64::__cuda==11.0=0

Your installed CUDA driver is: 11.0

```

## 4. Resources

1. [^1]Installing Tensorflow with CUDA & cuDNN GPU support on Ubuntu 20.04 and charge through your Linear Algebra calculations from [here](https://medium.com/@paulrohan/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a)

2. Install TensorFlow with GPU Support documentation from [here](https://www.tensorflow.org/install/gpu)

3. How to install the NVIDIA drivers on Ubuntu 20.04 Focal Fossa Linux from [here](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)