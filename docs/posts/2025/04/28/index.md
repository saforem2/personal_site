# 🔥 Building PyTorch 2.6 from Source on Aurora
Sam Foreman
2025-04-28

<link rel="preconnect" href="https://fonts.googleapis.com">

- [PyTorch 2.6 on Aurora](#pytorch-26-on-aurora)

## PyTorch 2.6 on Aurora

``` bash
; source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env
; export CXX=$(which g++)
; export CC=$(which gcc)
; export REL_WITH_DEB_INFO=1
; export USE_CUDA=0
; export USE_ROCM=0
; export USE_MKLDNN=1
; export USE_MKL=1
; export USE_ROCM=0
; export USE_CUDNN=0
; export USE_FBGEMM=0
; export USE_NNPACK=0
; export USE_QNNPACK=0
; export USE_NCCL=0
; export USE_CUDA=0
; export BUILD_CAFFE2_OPS=0
; export BUILD_TEST=0
; export USE_DISTRIBUTED=1
; export USE_NUMA=0
; export USE_MPI=1
; export USE_XPU=1
; export USE_XCCL=1
; export INTEL_MKL_DIR=$MKLROOT
; export USE_AOT_DEVLIST='pvc'
; export TORCH_XPU_ARCH_LIST='pvc'
; export OCLOC_VERSION=24.39.1
; which -a gcc
# /opt/aurora/24.347.0/spack/unified/0.9.1/install/linux-sles15-x86_64/gcc-13.3.0/gcc-13.3.0-4enwbrb/bin/gcc
#/usr/bin/gcc
; which -a g++
#/opt/aurora/24.347.0/spack/unified/0.9.1/install/linux-sles15-x86_64/gcc-13.3.0/gcc-13.3.0-4enwbrb/bin/g++
#/usr/bin/g++
; git clone https://github.com/pytorch/pytorch.git
; cd pytorch
; git checkout v2.6.0
; git submodule update --init --recursive --force
; make triton
; USE_XCCL=1 DEBUG=1 REL_WITH_DEB_INFO=1 USE_CUDA=0 CC=$(which gcc) CXX=$(which g++) python3 setup.py bdist_wheel --verbose 2>&1 | tee "build_whl-${NOW}.log"
; python3 -m pip install dist/torch-2.6.0a0+git1eba9b3-cp310-cp310-linux_x86_64.whl --force-reinstall
; python -m pip install intel-extension-for-pytorch==2.6.10+xpu oneccl_bind_pt==2.6.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
