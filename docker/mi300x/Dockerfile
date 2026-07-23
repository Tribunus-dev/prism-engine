ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:latest
FROM ${ROCM_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:${PATH} \
    RUST_BACKTRACE=1 \
    PRISM_ROCM_GPU=gfx942 \
    HSA_ENABLE_SDMA=1 \
    PYTORCH_ROCM_ARCH=gfx942

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        git \
        libclang-dev \
        pkg-config \
        python3 \
        python3-pip \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /workspace/prism-engine
COPY . .

# Keep the AMD runtime as a standalone workspace, while exercising the
# top-level SpatialIR dependency through its manifest path.
# Keep the image build independent of Prism's optional model workspaces. The
# AMD runtime manifest is a standalone workspace and pulls in the native ECS
# and spatial IR crates through path dependencies.
RUN cargo test --manifest-path crates/prism-amd-npu-runtime/Cargo.toml --quiet

CMD ["bash"]
