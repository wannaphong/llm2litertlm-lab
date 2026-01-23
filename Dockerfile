# 1. Base Image: Official NVIDIA CUDA 12.1 devel image (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. System Setup & Dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# 3. Install Python 3.12 and Dev Tools
RUN apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-venv

# 4. Make Python 3.12 the default 'python' command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 5. Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 6. Install Base Colab Stack (PyTorch & Common Libs)
# We install these *before* your custom requirements to cache the heavy download.
# Note: We explicitly use the CUDA 12.1 index to prevent CPU-only torch issues.
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    jupyter \
    jupyterlab \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. Set Working Directory
WORKDIR /workspace

# 8. Fetch and Install requirements.txt from your GitHub Repo
# This pulls the latest file from the 'main' branch of 'llm2litertlm-lab'
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 9. Default Command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
