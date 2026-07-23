# Prism MI300X validation image

This image is a Linux/AMD64 ROCm build and validation environment for Prism. It validates the portable native XDNA compiler/runtime layers and Prism’s AMD GPU path on an MI300X host; MI300X does not provide Ryzen AI/XDNA hardware.

Build on macOS with Docker Desktop or Buildx:

```sh
docker build --platform linux/amd64 \
  -f docker/mi300x/Dockerfile \
  -t prism-mi300x:dev .
```

Run on an MI300X host with ROCm devices exposed:

```sh
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  prism-mi300x:dev
```

For the hardware gate, run the checked-in validator from the mounted source
tree so the packed ternary scorer is compiled and exercised on gfx942:

```sh
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e PRISM_MI300X_GPU=1 \
  -v "$PWD:/workspace/prism-engine" \
  -w /workspace/prism-engine \
  prism-mi300x:dev \
  bash docker/mi300x/validate.sh
```

The image intentionally does not install or emulate an amdxdna driver. Ryzen AI/XDNA hardware validation remains a separate Linux hardware gate.
