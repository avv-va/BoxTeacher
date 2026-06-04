#!/usr/bin/env bash
# Build and run the BoxTeacher Docker image.
#
#   docker/boxteacher.sh build          # build the image
#   docker/boxteacher.sh run            # drop into an interactive shell (default)
#   docker/boxteacher.sh run <cmd...>   # run a command in the container, e.g.:
#       docker/boxteacher.sh run python projects/BoxTeacher/train_net.py \
#           --config-file projects/BoxTeacher/configs/coco/boxteacher_phenobench_r50_1x.yaml --num-gpus 8
#
# The runtime flags below (GPU access, shm size, ulimits, dataset/weight/output
# mounts) are the reason this script exists -- they cannot live in the Dockerfile.
set -euo pipefail

IMAGE="boxteacher"
# Repo root = parent of this script's directory, regardless of where it's called.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmd="${1:-run}"
shift || true

case "$cmd" in
	build)
		docker build \
			-f "$ROOT/docker/boxteacher.Dockerfile" \
			--build-arg "USER_ID=$(id -u)" \
			-t "$IMAGE" \
			"$ROOT"
		;;
	run)
		# Create host dirs so the bind-mounts don't materialize as root-owned.
		# (incl. the phenobench image mountpoint, nested inside datasets/.)
		mkdir -p "$ROOT/datasets" "$ROOT/pretrained_models" "$ROOT/output" \
			"$ROOT/datasets/phenobench/images"
		# Source of the phenobench images (read from outside the repo). Override
		# with PHENOBENCH_IMAGES=... if the dataset lives elsewhere.
		PHENOBENCH_IMAGES="${PHENOBENCH_IMAGES:-/home/ava/data/phenobench-yolo/images}"
		# projects/ is bind-mounted so edits to the (pure-python) BoxTeacher
		# project code take effect live, with no image rebuild/recompile.
		# detectron2/ and AdelaiDet/ stay baked in (they hold compiled .so).
		docker run --rm -it \
			--gpus all \
			--shm-size=16g \
			--ulimit memlock=-1 --ulimit stack=67108864 \
			-v "$ROOT/datasets:/home/appuser/BoxTeacher/datasets" \
			-v "$ROOT/pretrained_models:/home/appuser/BoxTeacher/pretrained_models" \
			-v "$ROOT/output:/home/appuser/BoxTeacher/output" \
			-v "$ROOT/projects:/home/appuser/BoxTeacher/projects" \
			-v "$PHENOBENCH_IMAGES:/home/appuser/BoxTeacher/datasets/phenobench/images:ro" \
			"$IMAGE" \
			"${@:-/bin/bash}"
		;;
	*)
		echo "usage: $0 {build|run [cmd...]}" >&2
		exit 2
		;;
esac
