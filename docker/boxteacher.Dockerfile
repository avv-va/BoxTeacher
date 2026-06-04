# Dockerfile for building/running BoxTeacher (detectron2 + AdelaiDet + the
# BoxTeacher project) on the known-good 2022-era toolchain.
#
# Pins match the repo's original docker/Dockerfile: CUDA 11.1, torch 1.10,
# torchvision 0.11.1 (cu111). Do NOT bump these casually -- this codebase uses
# THC/THC.h and the old tensor.type() dispatch API, both removed in torch >=1.11.
#
# Build context must be the REPO ROOT (so the whole tree is COPY-able).
# Easiest is the helper script, which also sets the right run-time flags:
#   docker/boxteacher.sh build
#   docker/boxteacher.sh run
# Or build by hand:
#   docker build -f docker/boxteacher.Dockerfile -t boxteacher .

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# 20.04 (not the upstream image's 18.04) because it ships Python 3.8: the
# vendored detectron2 0.6 requires python>=3.7, and 18.04's Python 3.6 fails
# with "requires a different Python". torch 1.10 cu111 has cp38 wheels.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
		python3-opencv ca-certificates python3-dev python3-pip git wget sudo ninja-build \
	&& rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# Non-root user, mirroring the upstream Dockerfile.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
# Use apt's pip (Python 3.8) and upgrade it into the user site.
RUN python3 -m pip install --user --upgrade pip

# tensorboard/cmake/onnx first (cmake from apt is too old), then the pinned torch.
RUN pip install --user tensorboard cmake onnx
RUN pip install --user torch==1.10 torchvision==0.11.1 \
	-f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# No GPU is visible during `docker build`, so force CUDA extension compilation
# and declare the target arch(es). 8.0 = A100 (the GPU on this host); extra
# arches are included so the image is portable to other common GPUs.
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Bring in the repo (build/ and .venv/ are excluded via .dockerignore).
COPY --chown=appuser:sudo . /home/appuser/BoxTeacher
WORKDIR /home/appuser/BoxTeacher

# Fix the in-repo detectron2 setup.py: it references example projects
# (PointRend/DeepLab/Panoptic-DeepLab) that were trimmed from this repo, which
# makes `build develop` fail with "package directory ... does not exist".
# BoxTeacher does not use them, so drop those PROJECTS entries.
RUN sed -i \
	-e '/projects\/PointRend\/point_rend/d' \
	-e '/projects\/DeepLab\/deeplab/d' \
	-e '/projects\/Panoptic-DeepLab\/panoptic_deeplab/d' \
	setup.py

# Build detectron2 (repo root), then AdelaiDet, in editable/develop mode.
RUN rm -rf build **/*.so && pip install --user -e .
RUN cd AdelaiDet && rm -rf build **/*.so && pip install --user -e . && cd ..

# Pure-python runtime deps that setup.py doesn't pin/install. Placed after the
# C++ builds so editing them never invalidates the (slow) compile cache.
#  - Pillow<10: detectron2 0.6 uses Image.LINEAR, removed in Pillow 10.
#  - timm 0.6.x: the Swin backbone imports timm.models.layers, an API path
#    deprecated/removed in timm 1.x. Only listed in detectron2's [all] extra.
RUN pip install --user "Pillow<10" "timm==0.6.13"

# Fixed model cache, mirroring upstream.
ENV FVCORE_CACHE="/tmp"
# detectron2 looks for datasets under ./datasets or $DETECTRON2_DATASETS.
ENV DETECTRON2_DATASETS="/home/appuser/BoxTeacher/datasets"

# Train (mount datasets + pretrained_models at runtime; see boxteacher.sh):
#   python projects/BoxTeacher/train_net.py \
#     --config-file projects/BoxTeacher/configs/coco/boxteacher_r50_1x.yaml --num-gpus 8
# Eval:
#   python projects/BoxTeacher/train_net.py \
#     --config-file <cfg> --num-gpus 8 --eval-only MODEL.WEIGHTS <path/to/weights>
