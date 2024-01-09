FROM nvcr.io/nvidia/tensorrt:23.09-py3

SHELL ["/bin/bash", "-c"] 

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cppcheck \
    curl \
    git \
    libeigen3-dev \
    lsb-release \
    python3-dbg \
    python3-pip \
    sudo \
    libboost-all-dev \
    snapd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility,video,compute
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update \
    && apt-get install -y --no-install-recommends clang-format ssh \
    && rm -rf /var/lib/apt/lists/*