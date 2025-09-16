FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ca-certificates git curl nvtop tmux && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

RUN git clone --filter=blob:none https://github.com/btseytlin/any2json.git

WORKDIR /code/any2json

ENV UV_LINK_MODE=copy

RUN uv sync && \
    . .venv/bin/activate && \
    uv pip install -e . && \
    uv -q pip install vllm --torch-backend=auto

CMD ["bash"]
