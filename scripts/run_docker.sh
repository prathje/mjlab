#!/usr/bin/env sh
#
# Injects useful arguments for running mjlab in docker.
# See docs/source/installation.rst for usage.
#
# Patterned after the uv-in-docker example:
# https://github.com/astral-sh/uv-docker-example/blob/5748835918ec293d547bbe0e42df34e140aca1eb/run.sh
#
# Key arguments:
#   --rm                Remove the container after exiting
#   --runtime=nvidia    Use NVIDIA Container runtime to give GPU access
#                       (only injected when the toolkit is detected)
#   --gpus all          Expose all GPUs by default
#                       (only injected when the toolkit is detected)
#   --platform=...      Pinned to linux/amd64 on hosts whose Docker engine
#                       reports a different architecture (e.g. macOS on
#                       Apple Silicon). The mjlab uv lock only resolves for
#                       linux/x86_64, so a native arm64 build would fail
#                       wheel resolution. Rosetta/qemu emulation is slow
#                       but lets the demo run for evaluation.
#   -v .:/app           Mount current directory to /app (code changes don't require rebuild)
#   -v /app/.venv       Mount venv separately (keeps developer's environment out of container)
#   -p 8080:8080        Publish port 8080 for viewing mjlab web interface on the host
#   -it (conditional)   Launch in interactive mode if running in a terminal
#                       (Note: if running training, there's a blocking wandb prompt before training begins)
#   docker build        Build and launch the image (tag matches the Makefile)
#   "$@"                Forward all arguments to the docker image


if [ -t 1 ]; then
    INTERACTIVE="-it"
else
    INTERACTIVE=""
fi

# Auto-detect the NVIDIA Container Toolkit. Without this guard, the original
# `--runtime=nvidia --gpus all` flags hard-fail on hosts that lack the
# toolkit (e.g. macOS, or Linux machines without NVIDIA drivers installed).
GPU_FLAGS=""
runtimes=$(docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null)
case " $runtimes " in
    *" nvidia "*)
        GPU_FLAGS="--runtime=nvidia --gpus all"
        ;;
    *)
        echo "[run_docker.sh] NVIDIA Container Toolkit not detected; running CPU-only." >&2
        ;;
esac

# Pin the build/run platform to linux/amd64 when the local Docker engine
# reports a different architecture (e.g. linux/aarch64 on Apple Silicon).
# The uv lockfile only resolves wheels for linux x86_64 and darwin arm64,
# so a native arm64 Linux build would fail.
PLATFORM_FLAG=""
host_arch=$(docker info --format '{{.Architecture}}' 2>/dev/null)
case "$host_arch" in
    x86_64|amd64|"")
        ;;
    *)
        PLATFORM_FLAG="--platform=linux/amd64"
        echo "[run_docker.sh] Docker engine reports ${host_arch}; forcing linux/amd64 (emulated)." >&2
        ;;
esac

IMAGE_ID=$(docker build $PLATFORM_FLAG -qt mjlab .) || exit $?

docker run \
    --rm \
    $GPU_FLAGS \
    $PLATFORM_FLAG \
    --volume .:/app \
    --volume /app/.venv \
    --publish 8080:8080 \
    $INTERACTIVE \
    "$IMAGE_ID" \
    "$@"
