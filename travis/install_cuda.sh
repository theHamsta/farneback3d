 # install repo packages
CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
rm $CUDA_REPO_PKG

if $WITH_CUDNN ; then
ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG
dpkg -i $ML_REPO_PKG
fi

# update package lists
apt-get -y update

# install packages
CUDA_PKG_VERSION="7-5"
CUDA_VERSION="7.5"
apt-get install -y --no-install-recommends \
cuda-core-$CUDA_PKG_VERSION \
cuda-cudart-dev-$CUDA_PKG_VERSION \
cuda-cublas-dev-$CUDA_PKG_VERSION \
cuda-curand-dev-$CUDA_PKG_VERSION
# manually create CUDA symlink
ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

if $WITH_CUDNN ; then
	apt-get install -y --no-install-recommends libcudnn7-dev
fi
